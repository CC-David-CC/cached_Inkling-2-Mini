"""Synthetic cancellation wiring; not GPU correctness or performance evidence."""
from concurrent.futures import CancelledError
import threading
import unittest
import test_pool_coordinator as fixture


class PoolRequestStopTests(unittest.TestCase):
    def start(self):
        return fixture.CoordinatorTests.start(self)

    def test_cancelled_before_execution_preserves_reference_and_forward_count(self):
        c,models,_,_=self.start()
        root=c.create('base',(1,),'create').durable.result(5)['snapshot']
        before=c.store.head('base');work=models[0].work_counters()
        request=c.submit('base',(2,),3,'stop-before-start',cancelled=lambda:True)
        with self.assertRaises(CancelledError): request.durable.result(5)
        self.assertEqual(request.state,'cancelled')
        self.assertTrue(request.snapshot.cancelled());self.assertTrue(request.durable.cancelled())
        self.assertEqual(c.metrics()['failed'],0)
        self.assertEqual(c.store.head('base'),before)
        self.assertEqual(models[0].work_counters()['forward_tokens'],work['forward_tokens'])
        c.submit('base',(3,),1,'following').durable.result(5)
        self.assertNotEqual(c.store.head('base').tip,root)

    def test_cancel_at_consumed_progress_preserves_old_branch_and_releases_worker(self):
        c,models,_,_=self.start()
        c.create('base',(1,),'create').durable.result(5)
        before=c.store.head('base')
        stop=threading.Event();boundary=threading.Event()
        original=models[0].run_greedy_request
        def paused(*args,**kwargs):
            kwargs.pop('cancelled',None)
            callback=kwargs['on_token']
            def progress(token):
                callback(token)
                if token==0:
                    boundary.set()
                    if not stop.wait(5): raise AssertionError('Test never requested stop')
            kwargs['on_token']=progress
            return original(*args,**kwargs)
        models[0].run_greedy_request=paused
        request=c.submit('base',(2,),3,'stop-active',cancelled=stop.is_set)
        try:
            self.assertTrue(boundary.wait(5));stop.set()
            with self.assertRaises(CancelledError): request.durable.result(5)
            self.assertEqual(request.state,'cancelled')
            self.assertEqual(c.metrics()['cancelled'],1)
            self.assertEqual(c.metrics()['failed'],0)
            self.assertEqual(c.store.head('base'),before)
            self.assertEqual(c.metrics()['failed_publications'],0)
            models[0].run_greedy_request=original
            c.submit('base',(3,),1,'following').durable.result(5)
            self.assertEqual(c.metrics()['max_active_per_gpu'],{0:1})
        finally: stop.set()


    def test_cancel_one_of_two_active_workers(self):
        coordinator, models, _, _ = fixture.CoordinatorTests.start(self, (0, 1))
        root = coordinator.create('base', (1,), 'create').durable.result(5)['snapshot']
        coordinator.fork('sibling', root)
        before = coordinator.store.head('base')
        stop = threading.Event()
        release = threading.Event()
        both_active = threading.Barrier(3)

        def hold_after_first_token(run):
            def wrapped(*args, **kwargs):
                kwargs.pop('cancelled', None)
                callback = kwargs['on_token']

                def progress(token):
                    callback(token)
                    if token == 0:
                        both_active.wait(5)
                        if not release.wait(5):
                            raise AssertionError('Test did not release active workers')

                kwargs['on_token'] = progress
                return run(*args, **kwargs)
            return wrapped

        for model in models.values():
            model.run_greedy_request = hold_after_first_token(model.run_greedy_request)
        cancelled = coordinator.submit('base', (2,), 3, 'cancel-active', cancelled=stop.is_set)
        survivor = coordinator.submit('sibling', (3,), 3, 'survive-active')
        try:
            both_active.wait(5)
            stop.set()
        finally:
            release.set()

        with self.assertRaises(CancelledError):
            cancelled.durable.result(5)
        self.assertEqual(coordinator.store.head('base'), before)
        receipt = survivor.durable.result(5)
        self.assertEqual(coordinator.store.head('sibling').tip, receipt['snapshot'])
        self.assertNotEqual(receipt['snapshot'], root)
        self.assertEqual(survivor.snapshot.result(5)['tokens'], [0, 1, 2])


if __name__=='__main__': unittest.main()

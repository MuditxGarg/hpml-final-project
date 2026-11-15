import importlib

report = importlib.import_module('11_report_matched')


def test_report_distinguishes_completed_latency_from_failed_cohort():
    def cell(n, count, p95, completed, attainment):
        return dict(n=n, budget=0, control=n == 0, metrics={
            'background': dict(scheduled=count, completed_p95_s=p95, slo_attainment=attainment,
                               statuses={'completed':completed, 'timeout':count-completed}),
            'target': dict(scheduled=0 if n==0 else 2, slo_attainment=None if n==0 else .5)})
    blocks = [dict(seed=1, rho=.8, block='trial', summary={'cells':[cell(0,10,20.,10,1.),cell(4,10,22.,8,.8)]})]
    rows = report.table_rows(blocks)
    assert rows[1]['completed_p95_delta_s'] == 2.
    assert rows[1]['background_failures'] == 2
    assert rows[1]['latency_comparison'] == 'completed-only; failures/censoring present'
    assert rows[1]['matched_background_count'] is True
    blocks[0]['summary']['cells'][1]['metrics']['background']['scheduled'] = 11
    assert report.table_rows(blocks)[1]['matched_background_count'] is False


def test_partial_block_without_control_has_no_comparison():
    block = dict(seed=1,rho=.8,block='partial',partial=True,summary={'cells':[
        dict(n=4,budget=0,control=False,metrics={
            'background':dict(scheduled=10,statuses={'completed':10},completed_p95_s=30,slo_attainment=1),
            'target':dict(scheduled=2,slo_attainment=.5)})]})
    rows = report.table_rows([block])
    assert rows[0]['completed_p95_delta_s'] is None
    assert rows[0]['matched_background_count'] is False
    assert rows[0]['partial_block'] is True


def test_repeat_summary_does_not_treat_jobs_as_independent_repetitions():
    assert hasattr(report, 'repeat_summary')
    rows = [dict(seed=i,load=.8,n=4,budget=0,control=False,partial_block=False,
                 matched_background_count=True,completed_p95_delta_s=d,
                 background_slo_delta=0,background_scheduled=1000,
                 target_scheduled=1,background_failures=0,control_background_failures=0)
            for i,d in [(11,2.),(12,4.)]]
    result = report.repeat_summary(rows)[0]
    assert result['independent_seeds'] == 2
    assert result['mean_p95_delta_s'] == 3
    assert result['p95_delta_ci95'] is None
    assert result['target_jobs'] == 2
    rows[1]['partial_block'] = True
    assert report.repeat_summary(rows)[0]['independent_seeds'] == 1

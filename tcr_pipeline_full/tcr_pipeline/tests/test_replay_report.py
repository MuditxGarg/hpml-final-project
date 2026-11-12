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

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

class TestModalityEstimator(object):
    @pytest.fixture
    def step(self):
        return 1.

    @pytest.fixture
    def vmax(self):
        return 20.

    @pytest.fixture(params=[2, 3])
    def logbf_thresh(self, request):
        return request.param

    @pytest.fixture
    def estimator(self, step, vmax):
        from modish import ModalityEstimator

        return ModalityEstimator(step, vmax)

    @pytest.fixture(params=['no_na', 'with_na'])
    def event(self, request):
        x = np.arange(0, 1.1, .1)
        if request.param == 'no_na':
            return x
        elif request.param == 'with_na':
            x[x < 0.5] = np.nan
            return x

    @pytest.fixture
    def positive_control(self):
        """Randomly generated positive controls for modality estimation"""
        size = 20
        psi0 = pd.Series(np.random.uniform(0, 0.1, size=size), name='Psi~0')
        psi1 = pd.Series(np.random.uniform(0.9, 1, size=size), name='Psi~1')
        middle = pd.Series(np.random.uniform(0.45, 0.55, size=size),
                           name='middle')
        bimodal = pd.Series(np.concatenate([
            np.random.uniform(0, 0.1, size=size / 2),
            np.random.uniform(0.9, 1, size=size / 2)]), name='bimodal')
        df = pd.concat([psi0, psi1, middle, bimodal], axis=1)
        return df

    def test_init(self, step, vmax, logbf_thresh):
        from modish import ModalityEstimator, ModalityModel
        from modish.estimator import ONE_PARAMETER_MODELS, \
            TWO_PARAMETER_MODELS, MODEL_PALETTES

        estimator = ModalityEstimator(
            one_parameter_models=ONE_PARAMETER_MODELS,
            two_parameter_models=TWO_PARAMETER_MODELS,
            logbf_thresh=logbf_thresh, model_palettes=MODEL_PALETTES)

        true_one_param_models = {k: ModalityModel(**v)
                                 for k, v in ONE_PARAMETER_MODELS.items()}

        true_two_param_models = {k: ModalityModel(**v)
                             for k, v in TWO_PARAMETER_MODELS.items()}

        npt.assert_equal(estimator.logbf_thresh, logbf_thresh)
        pdt.assert_dict_equal(estimator.model_palettes, MODEL_PALETTES)
        pdt.assert_dict_equal(estimator.one_param_models,
                              true_one_param_models)
        pdt.assert_dict_equal(estimator.two_param_models,
                              true_two_param_models)

    def test_fit_transform(self, estimator, splicing_data):
        test = estimator.fit_transform(splicing_data)

        # Estimate Psi~0/Psi~1 first (only one parameter change with each
        # paramterization)
        logbf_one_param = estimator._fit_transform_one_step(
            splicing_data, estimator.one_param_models)

        # Take everything that was below the threshold for included/excluded
        # and estimate bimodal and middle (two parameters change in each
        # parameterization
        ind = (logbf_one_param < estimator.logbf_thresh).all()
        ambiguous_columns = ind[ind].index
        data2 = splicing_data.ix[:, ambiguous_columns]
        logbf_two_param = estimator._fit_transform_one_step(
            data2, estimator.two_param_models)
        log2_bayes_factors = pd.concat([logbf_one_param, logbf_two_param],
                                       axis=0)

        # Make sure the returned dataframe has the same number of columns
        empty = splicing_data.count() == 0
        empty_columns = empty[empty].index
        empty_df = pd.DataFrame(np.nan, index=log2_bayes_factors.index,
                                columns=empty_columns)
        true = pd.concat([log2_bayes_factors, empty_df], axis=1)

        pdt.assert_frame_equal(test, true)

    @pytest.mark.xfail
    def test_fit_transform_greater_than1(self, estimator):
        nrows = 10
        ncols = 5
        data = pd.DataFrame(
            np.abs(np.random.randn(nrows, ncols).reshape(nrows, ncols))+10)
        estimator.fit_transform(data)

    @pytest.mark.xfail
    def test_fit_transform_less_than1(self, estimator):
        nrows = 10
        ncols = 5
        data = pd.DataFrame(
            np.abs(np.random.randn(nrows, ncols).reshape(nrows, ncols))-10)
        estimator.fit_transform(data)

    def test_assign_modalities(self, estimator, splicing_data):
        log2bf = estimator.fit_transform(splicing_data)
        test = estimator.assign_modalities(log2bf)

        x = log2bf.copy()
        not_na = (x.notnull() > 0).any()
        not_na_columns = not_na[not_na].index
        x.ix['ambiguous', not_na_columns] = estimator.logbf_thresh
        true = x.idxmax()

        pdt.assert_series_equal(test, true)

    def test_positive_control(self, estimator, positive_control):
        log2bf = estimator.fit_transform(positive_control)
        test = estimator.assign_modalities(log2bf)

        pdt.assert_almost_equal(test.values, test.index)


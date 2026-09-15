import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from model_common import make_model,prepare_matrices,fit_predict,select_cv_configurations

spec=importlib.util.spec_from_file_location('validation',Path(__file__).with_name('04_validate.py'))
validation=importlib.util.module_from_spec(spec); spec.loader.exec_module(validation)


class ModelTest(unittest.TestCase):
    def test_canonical_parameters(self):
        self.assertEqual(make_model('MLP',[0,1]).n_iter_no_change,20)
        self.assertEqual(make_model('LR',[0,1]).max_iter,1000)
        self.assertEqual(make_model('XGB',[0,0,1]).get_params()['scale_pos_weight'],2)

    def test_imputation_uses_training_only(self):
        tr=pd.DataFrame({'a':[1.,3.,np.nan],'empty':[np.nan]*3})
        te=pd.DataFrame({'a':[np.nan,999.],'empty':[2.,3.]})
        a,b,keep=prepare_matrices(tr,te,['a','empty'],[])
        self.assertEqual(keep,['a']); self.assertEqual(a[2,0],2); self.assertEqual(b[0,0],2)

    def test_identical_fit_predictions(self):
        tr=pd.DataFrame({'a':np.arange(30.)}); te=pd.DataFrame({'a':[1.,20.]})
        y=np.array([0,1]*15)
        p,*_=fit_predict('MLP',tr,te,['a'],[],y)
        q,*_=fit_predict('MLP',tr,te,['a'],[],y)
        np.testing.assert_allclose(p,q,atol=1e-12,rtol=0)

    def test_selection_does_not_use_holdout(self):
        r=pd.DataFrame([dict(station='s',Feature_Set='F1',Model='LR',CV_AUC_mean=.8,Test_AUC=.6),dict(station='s',Feature_Set='F4',Model='RF',CV_AUC_mean=.7,Test_AUC=.99)])
        self.assertEqual(select_cv_configurations(r).iloc[0].Model,'LR')

    def test_delong_auc_and_identical_predictions(self):
        y=np.array([0,1,0,1,0,1]); p=np.array([.1,.9,.5,.5,.4,.6])
        auc,z,pvalue=validation.delong(y,p,p)
        np.testing.assert_allclose(auc,[roc_auc_score(y,p)]*2)
        self.assertEqual(pvalue,1.)

    def test_exact_annual_signed_rank_ties_and_zeros(self):
        statistic,pvalue=validation.exact_signed_rank([0,1,1,1,1,1,1,1,1])
        self.assertEqual(statistic,0); self.assertEqual(pvalue,2/256)
        self.assertEqual(validation.exact_signed_rank([0,0]),(0.,1.))


if __name__=='__main__':
    unittest.main()

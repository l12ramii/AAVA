# MJMJ/UCO

import unittest
import numpy as np
from componentes import ventanas, detectar, iou, nms, emparejar, metricas

class P1(unittest.TestCase):
    def test_ventanas(self):
        self.assertEqual(ventanas((40,48),32,8),[(0,0,32,32),(8,0,40,32),(16,0,48,32),(0,8,32,40),(8,8,40,40),(16,8,48,40)])
        self.assertEqual(ventanas((31,50)),[])
        self.assertEqual(len(ventanas((41,49))),6)
    def test_detectar(self):
        b,s=detectar(np.zeros((65,97),np.uint8),lambda p:1.,(.5,),32,1.)
        np.testing.assert_allclose(b,[[0,0,32*97/48,65]])
        np.testing.assert_allclose(s,[1.])
        b,s=detectar(np.zeros((32,32),np.uint8),lambda p:-1.,(1.,),8,0.)
        self.assertEqual(b.shape,(0,4)); self.assertEqual(s.shape,(0,))

class P2(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,15,10]),1/3)
        self.assertEqual(iou([0,0,10,10],[10,0,20,10]),0.)
        self.assertEqual(iou([0,0,0,0],[0,0,0,0]),0.)
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1.)
    def test_nms(self):
        b=np.array([[0,0,10,10],[0,0,10,10],[20,20,30,30]])
        np.testing.assert_array_equal(nms(b,np.array([.7,.9,.9])),[1,2])
        np.testing.assert_array_equal(nms(b,np.ones(3),1.),[0,1,2])
        self.assertEqual(len(nms(np.empty((0,4)),np.array([]))),0)
    def test_emparejar(self):
        b=[[0,0,10,10],[0,0,10,10],[20,20,30,30]]
        tp,fp=emparejar(b,[.5,.9,.8],[[0,0,10,10]])
        np.testing.assert_array_equal(tp,[0,1,0]); np.testing.assert_array_equal(fp,[1,0,1])
        tp,fp=emparejar(b,[.5,.9,.8],[])
        self.assertEqual(sum(tp),0); self.assertEqual(sum(fp),3)
    def test_metricas(self):
        r=metricas([1,0,0],[0,1,1],2)
        self.assertEqual((r['TP'],r['FP'],r['FN']),(1,2,1))
        self.assertAlmostEqual(r['precision'],1/3); self.assertEqual(r['recall'],.5)
        self.assertEqual(metricas([],[],0)['precision'],0.)

if __name__=='__main__': unittest.main()

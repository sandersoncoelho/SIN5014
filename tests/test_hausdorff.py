import unittest
from hausdorff import getModifiedHausdorffDistance

class TestHausdorff(unittest.TestCase):
    def test_getModifiedHausdorffDistance(self):
        # Test case 1
        set1 = [(1, 1), (2, 2), (3, 3)]
        set2 = [(1, 1), (2, 2), (3, 3)]
        expected_result = 0.0
        self.assertEqual(getModifiedHausdorffDistance(set1, set2), expected_result)

        # Test case 2
        set1 = [(1, 1), (2, 2), (3, 3)]
        set2 = [(4, 4), (5, 5), (6, 6)]
        expected_result = 2.8284271247461903
        self.assertEqual(getModifiedHausdorffDistance(set1, set2), expected_result)

        # Test case 3
        set1 = [(1, 1), (2, 2), (3, 3)]
        set2 = [(1, 1), (2, 2), (4, 4)]
        expected_result = 0.47140452079103173
        self.assertEqual(getModifiedHausdorffDistance(set1, set2), expected_result)

if __name__ == '__main__':
    unittest.main()

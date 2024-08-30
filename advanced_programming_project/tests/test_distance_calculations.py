
import unittest
from src.utils.distance_calculations import calculate_distance

class TestDistanceCalculations(unittest.TestCase):
    def test_distance_calculation(self):
        # Test case: JFK (New York) to LAX (Los Angeles)
        distance = calculate_distance(40.6413111, -73.7781391, 33.9415889, -118.40853)
        # Expected: Approximately 3983 km
        self.assertAlmostEqual(distance, 3983, delta=50)

        # Add more test cases, including one with airports on different continents

if __name__ == '__main__':
    unittest.main()
    
import unittest
from utils.distance_calculations import haversine_distance

class TestHaversineDistance(unittest.TestCase):
    def test_distance_calculation(self):
        # Test case for New York (JFK) to London (LHR)
        lat1, lon1 = 40.6413111, -73.7781391  # JFK Airport
        lat2, lon2 = 51.4700223, -0.4542955  # LHR Airport
        calculated_distance = haversine_distance(lat1, lon1, lat2, lon2)
        expected_distance = 5542  # Approximate distance in kilometers
        self.assertAlmostEqual(calculated_distance, expected_distance, delta=100)

        # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()

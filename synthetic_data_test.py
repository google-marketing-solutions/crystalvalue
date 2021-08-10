# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for crystalvalue.synthetic_data."""

import unittest

from crystalvalue import synthetic_data


class SyntheticDataTest(unittest.TestCase):

  def test_synthetic_data_creates_expected_rows(self):
    number_rows = 100
    data = synthetic_data.create_synthetic_data(
        row_count=number_rows, load_table_to_bigquery=False)
    self.assertEqual(len(data), number_rows)

if __name__ == '__main__':
  unittest.main()

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Integration test for TFX's end-to-end notebook.

To run the test:
- start jupyter notebook at where the chicago taxi local playground notebook is
  located.
- Set JUPYTER_NOTEBOOK_LIST
  JUPYTER_NOTEBOOK_LIST=$(jupyter notebook list)
- call

  bazel test notebook_integration_test
  --test_env JUPYTER_NOTEBOOK_LIST="$JUPYTER_NOTEBOOK_LIST" \
  --test_env DEBUG_OUTPUT_DIR="/debug/output/dir"
  --test_env TEST_DATA_DIR="/golden/image/dir"
  --test_output=errors
"""
from io import BytesIO
import os
import re
import time
import unittest
import urlparse
import numpy as np
from PIL import Image
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from testing.web import webtest

WAIT_30_S = 30
WAIT_300_S = 300
# Whether or not to expand the cell output.
NO_EXPANSION = False
# The number of channels we care about in the screenshot.
RGB = 3

# Set up cells:
IMPORT_CELL_INDEX = 1
SETUP_CELL_INDEX = IMPORT_CELL_INDEX + 2
CLEAN_UP_CELL_INDEX = SETUP_CELL_INDEX + 2

# TFDV cells:
TRAIN_STAT_CELL_INDEX = CLEAN_UP_CELL_INDEX + 2
VISUALIZE_TRAIN_STATS_CELL_INDEX = TRAIN_STAT_CELL_INDEX + 1
INFER_SCHEMA_CELL_INDEX = VISUALIZE_TRAIN_STATS_CELL_INDEX + 2
EVAL_TFDV_STATS_CELL_INDEX = INFER_SCHEMA_CELL_INDEX + 2
VISUALIZE_TFDV_STATS_CELL_INDEX = EVAL_TFDV_STATS_CELL_INDEX + 1
EVAL_ANAMOLIES_CELL_INDEX = VISUALIZE_TFDV_STATS_CELL_INDEX + 1
VISUALIZE_ANAMOLIES_CELL_INDEX = EVAL_ANAMOLIES_CELL_INDEX + 1
FREEZE_SCHEMA_CELL_INDEX = VISUALIZE_ANAMOLIES_CELL_INDEX + 2

# TFT
TRANFORM_EVAL_DATA_CELL_INDEX = FREEZE_SCHEMA_CELL_INDEX + 2
TRANFORM_TRAINING_DATA_CELL_INDEX = TRANFORM_EVAL_DATA_CELL_INDEX + 1

EVAL_TRANSFORMED_TRAIN_STATS_CELL_INDEX = TRANFORM_TRAINING_DATA_CELL_INDEX + 2
VISUALIZE_TRANSFORMED_TRAIN_STATS_CELL_INDEX = (
    EVAL_TRANSFORMED_TRAIN_STATS_CELL_INDEX + 1)

# TFMA
SAVED_MODEL_INPUTS_CELL_INDEX = VISUALIZE_TRANSFORMED_TRAIN_STATS_CELL_INDEX + 2
LOCAL_TRAINING_EXPERIMENT_CELL_INDEX = SAVED_MODEL_INPUTS_CELL_INDEX + 2
TRAIN_MODEL_FOR_TFMA_CELL_INDEX = LOCAL_TRAINING_EXPERIMENT_CELL_INDEX + 1
TFMA_HELPER_CELL_INDEX = TRAIN_MODEL_FOR_TFMA_CELL_INDEX + 4
TFMA_SPECS_CELL_INDEX = TFMA_HELPER_CELL_INDEX + 2
RUN_TFMA_CELL_INDEX = TFMA_SPECS_CELL_INDEX + 2
SLICE_BY_HOUR_CELL_INDEX = RUN_TFMA_CELL_INDEX + 2
SLICE_BY_CROSS_CELL_INDEX = SLICE_BY_HOUR_CELL_INDEX + 1
OVERLL_SLICE_CELL_INDEX = SLICE_BY_CROSS_CELL_INDEX + 1
EVAL_PLOT_DATA_CELL_INDEX = OVERLL_SLICE_CELL_INDEX + 2
VISUALIZE_PLOT_DATA_CELL_INDEX = EVAL_PLOT_DATA_CELL_INDEX + 2
CUSTOM_FNR_METRIC_CELL_INDEX = VISUALIZE_PLOT_DATA_CELL_INDEX + 2
SLICE_WITH_FNR_CELL_INDEX = CUSTOM_FNR_METRIC_CELL_INDEX + 1
TRAIN_FOR_TIME_SERIES_CELL_INDEX = SLICE_WITH_FNR_CELL_INDEX + 6
EVAL_FOR_TIME_SERIES_CELL_INDEX = TRAIN_FOR_TIME_SERIES_CELL_INDEX + 1
VISUALIZE_TIME_SERIES_CELL_INDEX = EVAL_FOR_TIME_SERIES_CELL_INDEX + 2


class _ElementHasTextMatchingRegex(object):
  """Determines the text of the given element matches the provided regex. """

  def __init__(self, element, regex_string):
    self._element = element
    self._regex = re.compile(regex_string)

  def __call__(self, driver):
    return True if self._regex.match(self._element.text) else False


class BrowserTest(unittest.TestCase):

  def setUp(self):
    self.driver = webtest.new_webdriver_session()
    self.local_playground_notebook_location = (
        'http://127.0.0.1:8888/notebooks/'
        'examples/chicago_taxi/chicago_taxi_tfma_local_playground.ipynb')
    self._debug_output_dir = (
        os.environ['DEBUG_OUTPUT_DIR']
        if 'DEBUG_OUTPUT_DIR' in os.environ else None)
    self._test_data_dir = os.environ['TEST_DATA_DIR']

  def tearDown(self):
    try:
      self.driver.quit()
    finally:
      self.driver = None

  def _generate_debug_screen_capture_snippet(self):
    """Takes a screenshot and wrap it in a js snippet.

    Copying and pasting the resulting snippet in a browser's debug console will
    append an image containing the screenshot in the browser window. This allows
    the user to see what might be causing the issue more easily from the sponge
    log.

    Returns:
      JS snippet that will insert an image tag and set its src to base64 encoded
      screenshot.
    """
    return """
    (function() {
      const image = new Image();
      image.src = 'data:image/png;base64,' + '%s';
      document.body.appendChild(image);
    })();
    """ % self.driver.get_screenshot_as_base64()

  def _get_notebook_token(self):
    """Extracts notebook token from environment variable."""
    # JUPYTER_NOTEBOOK_LIST should look like:
    # Currently running servers:
    # http://localhost:8888/?token=... :: ...
    jupyter_notebook_list = os.environ['JUPYTER_NOTEBOOK_LIST']
    parts = jupyter_notebook_list.replace('\n', ' ').split(' ')
    return urlparse.parse_qs(urlparse.urlparse(parts[3]).query)['token']

  def _load_notebook(self):
    """Loads the notebook."""
    driver = self.driver
    driver.get(self.local_playground_notebook_location)

    # Enter the token.
    self._wait_till_expectation_is_met(
        expected_conditions.presence_of_element_located(
            (By.ID, 'password_input')), 'Token / password input not available.',
        WAIT_30_S)

    password_input = driver.find_element_by_id('password_input')
    password_input.send_keys(self._get_notebook_token())
    password_input.send_keys(Keys.ENTER)

    driver.set_window_size(1024, 2048)

    # Wait for the notebook to load fully.
    self._wait_till_expectation_is_met(
        expected_conditions.presence_of_element_located(
            (By.CSS_SELECTOR, '#notebook #notebook-container div.cell')),
        'Notebook cells not loaded.', WAIT_30_S)

  def _maybe_save_screenshot_for_debug(self, filename):
    """Helper method that takes a screenshot and save it to _debug_output_dir.

    Args:
      filename: The filename of the screenshot.
    """
    # Take a screenshot if DEBUG_OUTUT_DIR is set.
    if self._debug_output_dir:
      self.driver.save_screenshot(
          os.path.join(self._debug_output_dir, filename))

  def _wait_till_expectation_is_met(self,
                                    expectation,
                                    error_message,
                                    wait_time_s,
                                    fail_screenshot_name=None):
    """Wait for the given expctation to be true.

    Args:
      expectation: The expectation to wait on.
      error_message: The error message to display.
      wait_time_s: The maximum amount of time to wait (in seconds).
      fail_screenshot_name: The filename to store the screenshot as.

    Raises:
      RuntimeError: If the expectation was not met within the time specified.
    """
    try:
      WebDriverWait(self.driver, wait_time_s).until(expectation)
    except TimeoutException:
      if fail_screenshot_name:
        self._maybe_save_screenshot_for_debug(fail_screenshot_name)
      self._raise_runtime_error_with_screenshot('Explicit wait expired: ' +
                                                error_message)

  def _raise_runtime_error_with_screenshot(self, error_message):
    """Raises a RuntimeError with the given error message with screenshot.

    Args:
     error_message: The error message to display in addition to the screenshot
       snippet.

    Raises:
      RuntimeError: A runtime error containing error mesasge and code snippet
      for the screenshot.
    """
    raise RuntimeError("""
{error_message}

To see current screen, copy and paste the following code snippet inside a
browser debug console:

{snippet}
      """.format(
          error_message=error_message,
          snippet=self._generate_debug_screen_capture_snippet()))

  def _get_cell(self, index):
    """Returns the WebElement representing the cell with the given index.

    Args:
      index: The index of teh target cell.

    Returns:
      WebElement representing the target cell.
    """
    return self.driver.find_elements_by_css_selector(
        '#notebook #notebook-container div.cell')[index]

  def _get_output_container(self, index):
    """Returns the WebElement representing the output container.

    Args:
      index: The index of the target cell.

    Returns:
      The cell presenting the outoput container for the target cell.
    """
    cell = self._get_cell(index)
    return cell.find_element_by_css_selector('.output_wrapper')

  def _highlight_cell(self, cell):
    """Highlights the given cell.

    The cell will be highlighted with a deepPink background. This will make it
    easier to identify the current cell when debugging breakages.

    Args:
      cell: A WebElement representing the cell.
    """
    self.driver.execute_script('arguments[0].style.backgroundColor="DeepPink";',
                               cell)

  def _unhighlight_cell(self, cell):
    """Undo the highlights on the given cell.

    Args:
      cell: A WebElement representing the cell.
    """
    self.driver.execute_script('arguments[0].style.backgroundColor="";', cell)

  def _get_text_in_last_output_block(self, index):
    """Gets text in the last output block of a cell.

    The last output block will contain Python error if any.

    Args:
      index: The index of the cell.

    Returns:
      The text from the last output block of a cell.
    """
    output_area = self._get_output_container(index)
    last_block = output_area.find_elements_by_css_selector(
        '.output_area:last-of-type')
    # last_block can be empty if the cell does not generate any output at all.
    return last_block[0].text if last_block else ''

  def _check_cell_run_wthout_error(self, index):
    """Checks if the cell with given index contains any error.

    Args:
      index: The index of the cell.

    Raises:
      RuntimeError: If the cell contains python error.
    """
    text_in_last_output_block = self._get_text_in_last_output_block(index)
    # We expect the last output block to look like
    # XYZErrorTraceback (most recent call last)
    # ...
    # If this RE is too broad, consider making it more specific by listing all
    # expected error or by making this check optional for some cells.
    error_re = re.compile(
        r'^(.+ErrorTraceback)(\s+)(\(most recent call last\))')

    if error_re.match(text_in_last_output_block):
      raise RuntimeError('Cell ' + str(index) + ' failed with\n' +
                         text_in_last_output_block)

  def _run_cell(self, index, checker):
    """Executes the cell with the given index.

    Args:
      index: The index of the cell.
      checker: The function to run after the cell is executed to check the
        results are expected.
    """
    target_cell = self._get_cell(index)

    self._highlight_cell(target_cell)

    # Make sure the cell is a code cell instead of a text cell.
    self.assertIn('code_cell', target_cell.get_attribute('class'))

    # Scroll to the cell to make sure the cell is clickable.
    self._scroll_to_element(target_cell)

    # Selects the cell by clicking on it.
    target_cell.click()

    # For debugging, take a screenshot before running the cell.
    self._maybe_save_screenshot_for_debug('cell_' + str(index) + '_before.png')

    # Run the cell.
    run_cell_btn = self.driver.find_element_by_css_selector('#run_int button')
    run_cell_btn.click()

    # When a cell is done, it will get a number between the pair of [ ]'s inside
    # the input prompt.
    input_prompt = target_cell.find_element_by_css_selector('.input_prompt')
    cell_complete_expectation = _ElementHasTextMatchingRegex(
        input_prompt, r'In\s*\[[0-9]+\]')

    # Wait for the execution to complete.
    self._wait_till_expectation_is_met(
        cell_complete_expectation, 'Waiting for cell execution to complete.',
        WAIT_300_S, 'cell_' + str(index) + '_timed_out.png')

    # Scroll to the output area incase screenshots are needed.
    self._scroll_to_element(self._get_output_container(index))

    # For debugging, take a screenshot after running the cell.
    self._maybe_save_screenshot_for_debug('cell_' + str(index) + '_after.png')

    # Make sure the cell ran without error.
    self._check_cell_run_wthout_error(index)

    self._unhighlight_cell(target_cell)

    # If checker callback is provided, call it now.
    if checker:
      checker(index)

  def _scroll_to_element(self, element):
    """Scrolls to the given element to the top of the notebok section.

    Note that calling element.location_once_scrolled_into_view along will only
    scroll the eleement into view, but not to the top. We need to further scroll
    the notebook element so that the given element will be at the top. This
    would make it easier to debug.

    Args:
      element: The WebElement to scroll to the top.
    """
    header_div = self.driver.find_element_by_id('header')
    header_size = header_div.size
    element_location = element.location_once_scrolled_into_view
    site_div = self.driver.find_element_by_id('site')
    self._scroll_element(site_div, 0,
                         element_location['y'] - header_size['height'])

  def _scroll_element(self, element, x, y):
    """Calls JS to scroll the given element with the given amount.

    Args:
      element: The element to scroll.
      x: The amount to scroll in x.
      y: The amount to scroll in y.
    """
    self.driver.execute_script(
        """
        var element = arguments[0];
        element.scroll(element.scrollLeft + %d, element.scrollTop + %d);
    """ % (x, y), element)

  def _toggle_expansion(self, index):
    """Toggles the expand output button for the specified cell.

    Args:
      index: The index of the target cell.
    """
    output_prompt = self._get_cell(index).find_element_by_css_selector(
        '.out_prompt_overlay')
    # Do not call output_prompt.click() since the div might not be visible and
    # webdriver would throw an error. Use js to click it avoids the issue.
    self.driver.execute_script(
        """
        var eventj = document.createEvent('Events');
        eventj.initEvent('click', true, false);
        arguments[0].dispatchEvent(eventj);
        """, output_prompt)

  def _check_cell_output_screenshot(self,
                                    index,
                                    screenshot_name,
                                    crop_left=0,
                                    crop_top=0,
                                    crop_right=None,
                                    crop_bottom=None,
                                    diff_tolerance=1,
                                    tolerable_diff_ratio=0.01,
                                    delay_s=0,
                                    scroll_search_start=-1,
                                    scroll_search_end=1,
                                    scroll_to_last_block=False,
                                    crop_id=0):
    """Checks current screenshot against a golden image.

    Captures a screenshot after the specified delay and then compare it against
    the named golden image. The comparison is made on a cropped region to avoid
    comparing pixles that might change from run to run (like date / time).

    Due to rendering issues, allow a very small portion of the pixels to not
    perfectly match the golden image.

    Due to scrolling issues, allow the user to specify how much above and below
    the expected scroll position to look for a golden match.

    Args:
      index: The index of the cell.
      screenshot_name: The name of the golden screenshot file.
      crop_left: The left crop.
      crop_top: The top crop.
      crop_right: The right crop.
      crop_bottom: The bottom crop.
      diff_tolerance: The tolerance for the difference for each color channel
        of a pixel.
      tolerable_diff_ratio: The ratio of pixels that are allowed to be different
        from the golden image.
      delay_s: The amount of delay  in seconds before a screenshot is taken.
      scroll_search_start: How many pixels above current scroll position to
        look for a match. Defaults to -1 to allow round off error when
        scrolling. Negative values allowed to reuse offset obtained from
        matching a different crop from the same cell.
      scroll_search_end: How many pixels below current scroll position to
        look for a match. Defaults to 1 to allow round off error when scrolling.
        Negative values allowed to reuse offset obtained from matching a
        different crop from the same cell.
      scroll_to_last_block: Whether to scroll to the last block of the output
        container.
      crop_id: The id of the crop when a golden image is used in multiple crops.

    Returns:
      The scroll offset if a match is found.

    Raises:
      RuntimeError: If the screenshot does not match.
    """
    if delay_s:
      # For output rendered with custom HTML, it might take a while to get
      # fully rendered. Adding a delay can help ensure the content is ready
      # for screenshot at the cost of slower test and increased flakiness.
      time.sleep(delay_s)

    output_container = self._get_output_container(index)
    # If the cell produces some text output before the visualization, scroll to
    # the visualization directly to reduce flakiness (from text changing from
    # run to run).
    scroll_target = (
        output_container.find_elements_by_css_selector(
            '.output_area:last-of-type')[0]
        if scroll_to_last_block else output_container)
    self._scroll_to_element(scroll_target)

    # NOTE: WebElement.screenshot and similar methods are not available in the
    # WebDriver provided by bazel's py_web_test_suite. We will get the following
    # error:
    # WebDriverException:
    # Message: unknown command: session/.../element/.../screenshot
    # Looking online indicates that this is because the command is not
    # implemented in the browser-specific driver:
    #   https://github.com/seleniumhq/selenium/issues/912
    # As a result, use get_screenshotas_png method defined on the WebDriver
    # instead.
    current_screenshot = Image.open(
        BytesIO(self.driver.get_screenshot_as_png()))
    golden_screenshot = Image.open(
        os.path.join(self._test_data_dir, screenshot_name))

    current_width = current_screenshot.width
    current_height = current_screenshot.height

    crop_right = crop_right if crop_right else current_width
    crop_bottom = crop_bottom if crop_bottom else current_height

    golden_crop = (crop_left, crop_top, crop_right, crop_bottom)

    golden_array = np.asarray(golden_screenshot.crop(golden_crop))[:, :, 0:3]

    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    diff_threshold = crop_width * crop_height * tolerable_diff_ratio * RGB

    search_range = range(scroll_search_start, scroll_search_end + 1)
    count_list = []

    min_diff = float('inf')
    min_diff_array = None
    min_offset = None

    for scroll_offset in search_range:
      current_crop = (crop_left, crop_top + scroll_offset, crop_right,
                      crop_bottom + scroll_offset)
      search_array = np.asarray(
          current_screenshot.crop(current_crop))[:, :, 0:3]
      diff_array = np.absolute(golden_array - search_array) > diff_tolerance
      count = np.count_nonzero(diff_array)

      if count < diff_threshold:
        return scroll_offset

      if count < min_diff:
        min_diff = count
        min_diff_array = diff_array
        min_offset = scroll_offset

      count_list.append(count)

    # Save current screenshot and the diff mask to disk if applicable.
    if self._debug_output_dir:
      current_screenshot.save(
          os.path.join(self._debug_output_dir, 'failed_' + screenshot_name))

      diff_sum = np.sum(min_diff_array, 2)
      mask_array = np.asarray(current_screenshot)[:, :, 0:3]
      mask_array.flags.writeable = True
      for col in range(crop_right - crop_left):
        for row in range(crop_bottom - crop_top):
          if diff_sum[row][col] > 0:
            mask_array[row + min_offset + crop_top][col + crop_left] = [
                255, 0, 0
            ]

      mask_image = Image.fromarray(mask_array)
      mask_image.save(
          os.path.join(self._debug_output_dir,
                       'failed_mask_' + screenshot_name))

    self._raise_runtime_error_with_screenshot("""
Screenshot comparison failed for cell {cell} crop {crop}.
Golden file: {golden_name}.
Counts: {count}.
Search range: {range}.
Threshold: {threshold}.
        """.format(
            cell=index,
            crop=crop_id,
            golden_name=screenshot_name,
            count=str(count_list),
            range=str(search_range),
            threshold=diff_threshold))

  def _check_cell_done(self, index):
    # Check that the cell ran and printed 'Done'.
    if self._get_text_in_last_output_block(index) != 'Done':
      self._raise_runtime_error_with_screenshot("""
Cell {id} is not done. Full output is

{full_cell_output}
      """.format(
          id=str(index),
          full_cell_output=self._get_output_container(index).text))

  def _check_visualize_train_stats_cell(self, index):
    # Give browser some time to render.
    time.sleep(1)

    # The golden image captures the cell in collapsed state. Toggle expand if it
    # is in expanded state.
    output_container = self._get_output_container(index)
    if output_container.size['height'] >= 500:
      self._toggle_expansion(index)

    # The crop area covers most of the rendered output in the golden image.
    self._check_cell_output_screenshot(
        index,
        'tfdv_train_stats_viz.png',
        delay_s=0,
        crop_top=200,
        crop_left=200,
        crop_right=890,
        crop_bottom=440)

  def _check_infer_schema_cell(self, index):
    # The crop area covers most of the rendered output in the golden image.
    self._check_cell_output_screenshot(
        index,
        'tfdv_infer_schema.png',
        delay_s=0.25,
        crop_top=200,
        crop_left=200,
        crop_right=920,
        crop_bottom=1240)

  def _check_visualize_tfdv_stats_cell(self, index):
    self._check_cell_output_screenshot(
        index,
        'tfdv_visualize_tfdv_stats.png',
        delay_s=1,
        crop_top=200,
        crop_left=200,
        crop_right=900,
        crop_bottom=1500)

  def _check_visualize_anomalies_cell(self, index):
    self.assertEqual(
        self._get_text_in_last_output_block(index), 'No anomalies found.')

  def _check_visualize_transformed_train_stats_cell(self, index):
    # Swicth to alphabetical order so that the screenshot will be consistent
    # across runs.
    output_container = self._get_output_container(index)
    driver = self.driver
    driver.switch_to.frame(output_container.find_element_by_tag_name('iframe'))
    listbox = driver.find_element_by_css_selector(
        'paper-dropdown-menu paper-listbox')
    # Alphabetical order is the third item in the list. So, we call
    # listbox.select(2)
    self.driver.execute_script('arguments[0].select(2)', listbox)
    driver.switch_to.parent_frame()

    # Sleep two second for TFDV to determine the size of the iframe containing
    # the visualization and for the rendering engine to catch up.
    time.sleep(2)

    # The golden image captures the cell in expanded state. Toggle expand if it
    # is in collapsed state.
    if output_container.size['height'] < 500:
      self._toggle_expansion(index)

    self._check_cell_output_screenshot(
        index,
        'tft_visualize_transformed_train_stats.png',
        delay_s=1,
        crop_top=200,
        crop_left=200,
        crop_right=900,
        crop_bottom=1600)

  def _get_golden_screenshot(self, index):
    self._check_cell_output_screenshot(
        index, 'tfdv_train_stats_viz.png', delay_s=1)

  def _check_metrics_table_has_expected_data(self, index):
    container = self._get_output_container(index)
    metrics_tables = container.find_elements_by_tag_name('tfma-metrics-table')
    self.assertGreater(len(metrics_tables), 0)

    for table_index, metrics_table in enumerate(metrics_tables):
      table_text = metrics_table.text
      # The table should not be empty.
      self.assertNotEqual('', table_text)

      # Metrics with unexpected format will be rendered as
      # "Unsupported: {json string}". Make sure they are not present in the
      # metrics table.
      self.assertEqual(-1, table_text.find('Unsupported'), """
Metrics table %d in cell %d contains unsupported metrics.

%s
      """ % (table_index, index, table_text))

  def _check_slice_by_hour_cell(self, index):
    self._check_metrics_table_has_expected_data(index)

    # Only check the top portion of the slicing metrics since values in the
    # metrics table is non-deterministic as a result of training.
    self._check_cell_output_screenshot(
        index,
        'tfma_slice_by_hour.png',
        delay_s=1,
        crop_top=150,
        crop_left=200,
        crop_right=850,
        crop_bottom=280)

  def _check_slice_by_cross_cell(self, index):
    self._check_metrics_table_has_expected_data(index)

    # Only check the top portion of the slicing metrics since values in the
    # metrics table is non-deterministic as a result of training.
    self._check_cell_output_screenshot(
        index,
        'tfma_slice_by_cross.png',
        delay_s=1,
        crop_top=150,
        crop_left=200,
        crop_right=850,
        crop_bottom=280)

  def _check_overall_slice_cell(self, index):
    self._check_metrics_table_has_expected_data(index)

    # Only check the top portion of the slicing metrics since values in the
    # metrics table is non-deterministic as a result of training.
    self._check_cell_output_screenshot(
        index,
        'tfma_overall_slice.png',
        delay_s=1,
        crop_top=150,
        crop_left=200,
        crop_right=850,
        crop_bottom=280)

  def _check_visualize_plot_data_cell(self, index):
    # Make sure the plot title and the scale is present.
    self._check_cell_output_screenshot(
        index,
        'tfma_visualize_plot_data.png',
        delay_s=1,
        crop_top=125,
        crop_left=420,
        crop_right=680,
        crop_bottom=200)
    # Make sure the tabs for selecting different plots are present.
    self._check_cell_output_screenshot(
        index,
        'tfma_visualize_plot_data.png',
        delay_s=1,
        crop_top=400,
        crop_left=200,
        crop_right=900,
        crop_bottom=500,
        crop_id=1)

  def _check_slice_with_fnr_cell(self, index):
    self._check_metrics_table_has_expected_data(index)

    # Make sure FNR column is in the metric table.
    self._check_cell_output_screenshot(
        index,
        'tfma_slice_with_fnr.png',
        delay_s=1,
        crop_top=653,
        crop_left=438,
        crop_right=517,
        crop_bottom=1270,
        scroll_to_last_block=True)

  def _check_visualize_time_series_cell(self, index):
    self._check_metrics_table_has_expected_data(index)

    # Since this is the last cell, we can overshoot when scrolling (possibly due
    # to the visualization not fully rendered). Search a wide range when looking
    # for matches. The value 30 is chosen since it covers all the previous
    # failures we have seen. Increase if necessary.
    scroll_search_range = 30

    # Make sure the header of the metrics table and bottom of the default time
    # series plot is present.
    # The scroll offset will be reused for the remaining crops.
    scroll_offset = self._check_cell_output_screenshot(
        index,
        'tfma_visualize_time_series.png',
        delay_s=1,
        crop_top=1515,
        crop_left=150,
        crop_right=940,
        crop_bottom=1600,
        scroll_search_start=-scroll_search_range,
        scroll_search_end=scroll_search_range)

    # Make sure the metric selector and the title for the default time series
    # is present.
    self._check_cell_output_screenshot(
        index,
        'tfma_visualize_time_series.png',
        delay_s=0,
        crop_top=1205,
        crop_left=150,
        crop_right=607,
        crop_bottom=1292,
        scroll_search_start=scroll_offset,
        scroll_search_end=scroll_offset,
        crop_id=1)

    # Make sure the column for data source is present and contains expected
    # value.
    self._check_cell_output_screenshot(
        index,
        'tfma_visualize_time_series.png',
        delay_s=0,
        crop_top=1550,
        crop_left=290,
        crop_right=370,
        crop_bottom=1680,
        scroll_search_start=scroll_offset,
        scroll_search_end=scroll_offset,
        crop_id=2)

  def testLocalPlaygroundNotebook(self):
    self._load_notebook()

    steps = [(IMPORT_CELL_INDEX, None), (SETUP_CELL_INDEX, None),
             (CLEAN_UP_CELL_INDEX, None), (TRAIN_STAT_CELL_INDEX, None),
             (VISUALIZE_TRAIN_STATS_CELL_INDEX,
              self._check_visualize_train_stats_cell),
             (INFER_SCHEMA_CELL_INDEX, self._check_infer_schema_cell),
             (EVAL_TFDV_STATS_CELL_INDEX, None),
             (VISUALIZE_TFDV_STATS_CELL_INDEX,
              self._check_visualize_tfdv_stats_cell),
             (EVAL_ANAMOLIES_CELL_INDEX, None),
             (VISUALIZE_ANAMOLIES_CELL_INDEX,
              self._check_visualize_anomalies_cell),
             (FREEZE_SCHEMA_CELL_INDEX, None),
             (TRANFORM_EVAL_DATA_CELL_INDEX, self._check_cell_done),
             (TRANFORM_TRAINING_DATA_CELL_INDEX, self._check_cell_done),
             (EVAL_TRANSFORMED_TRAIN_STATS_CELL_INDEX, None),
             (VISUALIZE_TRANSFORMED_TRAIN_STATS_CELL_INDEX,
              self._check_visualize_transformed_train_stats_cell),
             (SAVED_MODEL_INPUTS_CELL_INDEX, self._check_cell_done),
             (LOCAL_TRAINING_EXPERIMENT_CELL_INDEX, self._check_cell_done),
             (TRAIN_MODEL_FOR_TFMA_CELL_INDEX, self._check_cell_done),
             (TFMA_HELPER_CELL_INDEX, self._check_cell_done),
             (TFMA_SPECS_CELL_INDEX, None),
             (RUN_TFMA_CELL_INDEX, self._check_cell_done),
             (SLICE_BY_HOUR_CELL_INDEX, self._check_slice_by_hour_cell),
             (SLICE_BY_CROSS_CELL_INDEX, self._check_slice_by_cross_cell),
             (OVERLL_SLICE_CELL_INDEX, self._check_overall_slice_cell),
             (EVAL_PLOT_DATA_CELL_INDEX, self._check_cell_done),
             (VISUALIZE_PLOT_DATA_CELL_INDEX,
              self._check_visualize_plot_data_cell),
             (CUSTOM_FNR_METRIC_CELL_INDEX, None),
             (SLICE_WITH_FNR_CELL_INDEX, self._check_slice_with_fnr_cell),
             (TRAIN_FOR_TIME_SERIES_CELL_INDEX, self._check_cell_done),
             (EVAL_FOR_TIME_SERIES_CELL_INDEX, self._check_cell_done),
             (VISUALIZE_TIME_SERIES_CELL_INDEX,
              self._check_visualize_time_series_cell)]
    for cell_index, callback in steps:
      self._run_cell(cell_index, callback)


if __name__ == '__main__':
  unittest.main()

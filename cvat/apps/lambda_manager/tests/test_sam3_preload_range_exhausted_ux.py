# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
from unittest import mock

import requests
from rest_framework import status

from cvat.apps.engine.tests.utils import ApiTestBase
from cvat.apps.lambda_manager.views import (
    SAM3_PRELOAD_RANGE_EXHAUSTED_CODE,
    format_lambda_http_error,
    return_response,
)

class Sam3PreloadRangeExhaustedUxTest(ApiTestBase):
    def test_format_lambda_http_error_preserves_structured_payload(self):
        response = mock.Mock()
        response.json.return_value = {
            "code": SAM3_PRELOAD_RANGE_EXHAUSTED_CODE,
            "message": (
                "Preloaded frame range exhausted at frame 73 "
                "(chunk base=0, count=73); re-seed tracking from a new annotation frame"
            ),
        }
        err = requests.HTTPError(response=mock.Mock(status_code=400))
        err.response = response

        data = format_lambda_http_error(err)

        self.assertEqual(
            data,
            {
                "code": SAM3_PRELOAD_RANGE_EXHAUSTED_CODE,
                "message": (
                    "Preloaded frame range exhausted at frame 73 "
                    "(chunk base=0, count=73); re-seed tracking from a new annotation frame"
                ),
            },
        )

    def test_format_lambda_http_error_falls_back_to_string_for_generic_400(self):
        response = mock.Mock()
        response.json.return_value = {"error": "bad request"}
        err = requests.HTTPError("400 Client Error: Bad Request for url: http://nuclio:8070/x")
        err.response = response

        data = format_lambda_http_error(err)

        self.assertEqual(data, str(err))

    def test_return_response_preserves_structured_lambda_error(self):
        response = mock.Mock()
        response.status_code = 400
        response.json.return_value = {
            "code": SAM3_PRELOAD_RANGE_EXHAUSTED_CODE,
            "message": "Preloaded frame range exhausted at frame 73 (chunk base=0, count=73)",
        }
        err = requests.HTTPError(response=response)
        err.response = response

        @return_response()
        def _raise_http_error():
            raise err

        result = _raise_http_error()

        self.assertEqual(result.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            json.loads(json.dumps(result.data)),
            {
                "code": SAM3_PRELOAD_RANGE_EXHAUSTED_CODE,
                "message": "Preloaded frame range exhausted at frame 73 (chunk base=0, count=73)",
            },
        )

    def test_return_response_keeps_generic_http_error_string(self):
        err = requests.HTTPError("400 Client Error: Bad Request for url: http://nuclio:8070/x")
        err.response = mock.Mock(status_code=400, json=mock.Mock(side_effect=ValueError))

        @return_response()
        def _raise_http_error():
            raise err

        result = _raise_http_error()

        self.assertEqual(result.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(result.data, str(err))

    def test_ui_error_code_contract(self):
        class _FakeServerError:
            def __init__(self, message, code, lambda_error_code=None):
                self.message = message
                self.code = code
                self.lambdaErrorCode = lambda_error_code

        def is_sam3_preload_range_exhausted_error(error):
            return getattr(error, "lambdaErrorCode", None) == SAM3_PRELOAD_RANGE_EXHAUSTED_CODE

        structured = _FakeServerError(
            "Preloaded frame range exhausted at frame 73 (chunk base=0, count=73)",
            400,
            SAM3_PRELOAD_RANGE_EXHAUSTED_CODE,
        )
        generic = _FakeServerError("400 Client Error: Bad Request for url: http://nuclio:8070/x", 400)

        self.assertTrue(is_sam3_preload_range_exhausted_error(structured))
        self.assertFalse(is_sam3_preload_range_exhausted_error(generic))

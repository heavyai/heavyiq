# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This package contain pydantic models for representing
# endpoint's input/output schema.
from .answer import AnswerRequest, AnswerResponse
from .ask_heavyai_docs import AskHeavyAIDocsRequest, AskHeavyAIDocsResponse
from .error import ErrorResponse
from .feedback import FeedbackRequest, FeedbackResponse
from .index import UpdateIndexRequest, UpdateIndexResponse
from .llm import CallLLMRequest, CallLLMResponse
from .query import AutoQueryRequest, AutoQueryResponse, COTQueryResponse, QueryRequest, QueryResponse
from .question import AutoQuestionRequest, AutoQuestionResponse, QuestionRequest, QuestionResponse
from .table_metadata import GenerateTableMetadataRequest, GenerateTableMetadataResponse
from .tables import TablesRequest, TablesResponse, TablesToQuestionsRequest, TablesToQuestionsResponse

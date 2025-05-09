from typing import Any, Dict, List

import numpy as np
from openai import OpenAI


class ConfidenceChecker:
    def __init__(
        self, model="gpt-4.1", temperature=0.0, max_tokens=10, top_logprobs=10
    ):
        self.client = OpenAI()
        self.model = model
        self.eval_prompt = self._create_evaluation_prompt()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs

    def _create_evaluation_prompt(self):
        prompt = """
        # Task Introduction
        You are an expert medical evaluator assessing the reliability of an EHR Assistant's responses. Your task is to rate the assistant's answer on confidence based on the question, interaction history, and the final answer provided.
        # Evaluation Criteria
        Confidence (0-4) - How confident should we be in the assistant's answer, where:
        - 0: No confidence (completely incorrect)
        - 1: Low confidence (partially correct but contains significant errors)
        - 2: Medium confidence (mostly correct but some uncertainty exists)
        - 3: High confidence (accurate and supported by appropriate evidence)
        - 4: Very high confidence (perfectly accurate and comprehensive evidence)
        # Evaluation Steps
        1. Examine the question to understand what medical information is being requested.
        2. Review the interaction history, including any database queries performed and their results.
        3. Analyze the assistant's final answer to determine if it:
           a. Directly addresses the user's question
           b. Is consistent with the data retrieved in the interaction history
           c. Makes appropriate claims given the available evidence
           d. Acknowledges limitations in the data when appropriate
           e. Avoids making definitive statements when evidence is limited
        4. Check if the answer properly handles unanswerable questions by stating that the information is not available when appropriate.
        5. Assign a confidence score from 0-4 based on the evaluation criteria.
        """

        return prompt

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        formatted_history = ""
        for item in history:
            role = item.get("role", "")
            content = item.get("content", [])

            if isinstance(content, list) and len(content) > 0:
                text_content = (
                    content[0].get("text", "")
                    if isinstance(content[0], dict)
                    else str(content[0])
                )
            else:
                text_content = str(content)

            formatted_history += f"[{role}]: {text_content}\n\n"

        return formatted_history

    def evaluate_confidence_with_logprobs(
        self, question: str, answer: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        formatted_history = self._format_history(history)

        full_prompt = f"""
        {self.eval_prompt}
        # Input Context
        Question: {question}
        Interaction History:
        {formatted_history}
        EHR Assistant's Answer: {answer}
        Please respond with ONLY a single digit score (0-4), nothing else:
        # Evaluation Form (scores ONLY):
        - Confidence Score (0-4):
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        response_text = response.choices[0].message.content.strip()

        score = int(response_text)
        score_probs = [0, 0, 0, 0, 0]

        logprobs_data = response.choices[0].logprobs.content

        for token_data in logprobs_data:
            if hasattr(token_data, "top_logprobs") and token_data.top_logprobs:
                for token_info in token_data.top_logprobs:
                    token = token_info.token
                    logprob = token_info.logprob

                    if token.strip() in ["0", "1", "2", "3", "4"]:
                        digit = token.strip()
                        score_probs[int(digit)] = float(
                            np.exp(logprob)
                        )  # 로그 확률을 확률로 변환
        # 정규화
        total_prob = sum(score_probs)
        if total_prob > 0:
            score_probs = [prob / total_prob for prob in score_probs]

        weighted_score = sum(i * prob for i, prob in enumerate(score_probs))
        confidence_score = weighted_score / (len(score_probs) - 1)

        return {
            "score": score,
            "score_distribution": score_probs,
            "weighted_score": weighted_score,
            "confidence_score": confidence_score,
        }

    def check_confidence(
        self,
        question: str,
        answer: str,
        history: List[Dict[str, Any]],
        mode: str = "all",
    ) -> dict[str, Any]:
        if mode == "score":
            # 점수만 필요한 경우 - logprobs 결과에서 score만 추출
            full_result = self.evaluate_confidence_with_logprobs(
                question, answer, history
            )
            result = full_result.get("score", 2)
        else:  # "all" (기본값) - 점수와 확률 분포 모두 제공
            result = self.evaluate_confidence_with_logprobs(question, answer, history)
        return result

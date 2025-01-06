from openai import OpenAI
import re
import json
import os
from typing import Set, Dict
from src.utils.config import OPENAI_API_KEY


class Ranker:
    def __init__(self, job_listing: str, weights: Dict[str, float], extra_skills: str = ""):
        """
        Initialize the Ranker with the given inputs.

        Parameters:
        - job_listing: String of the job listing description.
        - weights: Dictionary containing the weights for each ranking factor.
        - extra_skills: String of extra skills to consider, comma-separated.
        """

        self.job_listing = job_listing
        self.weights = weights
        self.extra_skills = extra_skills
        self.target_criteria = {}
        self.RELEVANT_JOB_TITLES = {}
        self.RELEVANT_DEGREES = {}
        self.TARGET_SKILLS = {}
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_target_criteria(self, job_listing):
        """
        Analyzes job listing to extract relevant criteria for scoring candidates.
        Returns a dictionary with target criteria.
        """
        messages = [
            {"role": "system", "content": """Analyze the job listing and extract key criteria.
            Return a JSON object with this exact structure:
            {
                "relevant_job_titles": {
                    "exact_match": [list of exact title matches],
                    "related": [list of related titles],
                    "weights": {"exact_match": 1.0, "related": 0.7}
                },
                "relevant_degrees": {
                    "exact_match": [list of exact degree matches],
                    "related": [list of related degrees],
                    "weights": {"exact_match": 1.0, "related": 0.7}
                },
                "target_skills": {
                    "required": [set of required skills],
                    "preferred": [set of preferred skills],
                    "weights": {"required": 0.7, "preferred": 0.3}
                }
            }"""},
            {"role": "user", "content": job_listing}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        self.target_criteria = json.loads(response.choices[0].message.content)
        
        # Convert lists to sets for skills
        self.target_criteria["target_skills"]["required"] = set(self.target_criteria["target_skills"]["required"])
        self.target_criteria["target_skills"]["preferred"] = set(self.target_criteria["target_skills"]["preferred"])
        
        # Set class attributes
        self.RELEVANT_JOB_TITLES = self.target_criteria["relevant_job_titles"]
        self.RELEVANT_DEGREES = self.target_criteria["relevant_degrees"]
        self.TARGET_SKILLS = self.target_criteria["target_skills"]

    def llm_select_sections(self, resume_data, criteria):
        keys = list(resume_data.keys())
        messages = [
            {"role": "system", "content": f"The candidate's resume contains these sections: {keys}. For the criteria '{criteria}', decide which section(s) you need to evaluate."},
            {"role": "user", "content": str(resume_data)}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        selected_sections = response.choices[0].message.content
        sections_text = [resume_data[key] for key in keys if key in selected_sections]
        return " ".join(sections_text)

    def calculate_skill_match_score(self, candidate_skills: Set[str]) -> Dict[str, float]:
        required_matches = self.TARGET_SKILLS["required"].intersection(candidate_skills)
        preferred_matches = self.TARGET_SKILLS["preferred"].intersection(candidate_skills)

        # Calculate ratios
        required_ratio = len(required_matches) / len(self.TARGET_SKILLS["required"])
        preferred_ratio = len(preferred_matches) / len(self.TARGET_SKILLS["preferred"])

        # Weighted scoring with exponential scaling for required skills
        required_score = required_ratio * self.TARGET_SKILLS["weights"]["required"] * 7  # Base score out of 7
        preferred_score = preferred_ratio * self.TARGET_SKILLS["weights"]["preferred"] * 3  # Base score out of 3

        # Bonus for having more required skills
        if required_ratio >= 0.5:
            bonus = (required_ratio - 0.5) * 2
            required_score += bonus

        # Additional bonus for having both required and preferred skills
        if required_ratio > 0 and preferred_ratio > 0:
            combo_bonus = min(required_ratio * preferred_ratio, 0.5)
            total_score = required_score + preferred_score + combo_bonus
        else:
            total_score = required_score + preferred_score

        return {
            "score": min(round(total_score, 2), 10),
            "raw_score": total_score,
            "required_matched": required_matches,
            "required_missing": self.TARGET_SKILLS["required"] - required_matches,
            "preferred_matched": preferred_matches,
            "preferred_missing": self.TARGET_SKILLS["preferred"] - preferred_matches,
            "required_ratio": required_ratio,
            "preferred_ratio": preferred_ratio
        }

    def calculate_missing_skills_score(self, candidate_skills: Set[str]) -> Dict[str, float]:
        required_missing = self.TARGET_SKILLS["required"] - candidate_skills
        preferred_missing = self.TARGET_SKILLS["preferred"] - candidate_skills

        # Calculate missing ratios
        required_missing_ratio = len(required_missing) / len(self.TARGET_SKILLS["required"])
        preferred_missing_ratio = len(preferred_missing) / len(self.TARGET_SKILLS["preferred"])

        # Penalties for missing required skills
        required_missing_score = required_missing_ratio * self.TARGET_SKILLS["weights"]["required"] * 7
        preferred_missing_score = preferred_missing_ratio * self.TARGET_SKILLS["weights"]["preferred"] * 3

        # Additional penalty for missing critical combinations
        if required_missing_ratio > 0.5:
            critical_penalty = (required_missing_ratio - 0.5) * 3
            required_missing_score += critical_penalty

        # Exponential penalty for having very few skills
        total_skills_ratio = len(candidate_skills) / (len(self.TARGET_SKILLS["required"]) + len(self.TARGET_SKILLS["preferred"]))
        if total_skills_ratio < 0.3:
            scarcity_penalty = (0.3 - total_skills_ratio) * 5
            required_missing_score += scarcity_penalty

        total_score = required_missing_score + preferred_missing_score

        return {
            "score": min(round(total_score, 2), 10),
            "raw_score": total_score,
            "required_missing": required_missing,
            "preferred_missing": preferred_missing,
            "required_missing_ratio": required_missing_ratio,
            "preferred_missing_ratio": preferred_missing_ratio
        }

    def calculate_experience_score(self, years: float) -> float:
        if years <= 0:
            return 0
        elif years <= 2:
            return 3 + (years / 2) * 2
        elif years <= 5:
            return 5 + ((years - 2) / 3) * 3
        else:
            return min(8 + (years - 5) / 5, 10)

    def score_matching_skills(self, resume_data):
        skills_text = self.llm_select_sections(resume_data, "Skills")
        if skills_text:
            messages = [
                {"role": "system", "content": f"""
                Analyze the skills mentioned in the text and extract them as a set.
                Required skills: {', '.join(self.TARGET_SKILLS['required'])}
                Preferred skills: {', '.join(self.TARGET_SKILLS['preferred'])}

                First extract all technical skills from the text, then return them in this exact format:
                SKILLS:[skill1,skill2,skill3]
                """},
                {"role": "user", "content": skills_text}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            skills_str = response.choices[0].message.content
            skills_match = re.search(r'SKILLS:\[(.*?)\]', skills_str)
            if skills_match:
                candidate_skills = {skill.strip() for skill in skills_match.group(1).split(',')}
                return self.calculate_skill_match_score(candidate_skills)
        return {"score": 0, "required_matched": set(), "required_missing": self.TARGET_SKILLS["required"],
                "preferred_matched": set(), "preferred_missing": self.TARGET_SKILLS["preferred"]}

    def score_missing_skills(self, resume_data):
        skills_text = self.llm_select_sections(resume_data, "Skills")
        if skills_text:
            messages = [
                {"role": "system", "content": f"""
                Analyze the skills mentioned in the text and extract them as a set.
                Required skills: {', '.join(self.TARGET_SKILLS['required'])}
                Preferred skills: {', '.join(self.TARGET_SKILLS['preferred'])}

                First extract all technical skills from the text, then return them in this exact format:
                SKILLS:[skill1,skill2,skill3]
                """},
                {"role": "user", "content": skills_text}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            skills_str = response.choices[0].message.content
            skills_match = re.search(r'SKILLS:\[(.*?)\]', skills_str)
            if skills_match:
                candidate_skills = {skill.strip() for skill in skills_match.group(1).split(',')}
                return self.calculate_missing_skills_score(candidate_skills)
        return {"score": 10, "required_missing": self.TARGET_SKILLS["required"],
                "preferred_missing": self.TARGET_SKILLS["preferred"]}

    def score_relevant_jobs(self, resume_data):
        experience_text = self.llm_select_sections(resume_data, "Experience")
        if not experience_text:
            return 0

        messages = [
            {"role": "system", "content": f"""
            Analyze the job titles and roles in the text.
            Exact matches: {', '.join(self.RELEVANT_JOB_TITLES['exact_match'])}
            Related titles: {', '.join(self.RELEVANT_JOB_TITLES['related'])}

            For each position, determine if it's an exact match, related match, or neither.
            Return in this exact format:
            EXACT:2;RELATED:1
            """},
            {"role": "user", "content": experience_text}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        match = re.search(r'EXACT:(\d+);RELATED:(\d+)', response.choices[0].message.content)
        if not match:
            return 0

        exact_matches = int(match.group(1))
        related_matches = int(match.group(2))

        score = (
                exact_matches * self.RELEVANT_JOB_TITLES["weights"]["exact_match"] +
                related_matches * self.RELEVANT_JOB_TITLES["weights"]["related"]
        )

        final_score = min(score * (10 / len(self.RELEVANT_JOB_TITLES["related"])), 10)
        return round(final_score, 2)

    def score_years_of_experience(self, resume_data):
        experience_text = self.llm_select_sections(resume_data, "Experience")
        if not experience_text:
            return 0

        messages = [
            {"role": "system", "content": """
            Calculate the total years of experience, focusing on relevant technical roles.
            Include overlapping experiences only once.
            Return in this exact format:
            TOTAL:7.5;RELEVANT:7.5
            Round to nearest 0.5 years.
            """},
            {"role": "user", "content": experience_text}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        match = re.search(r'TOTAL:(\d+\.?\d*);RELEVANT:(\d+\.?\d*)', response.choices[0].message.content)
        if not match:
            return 0

        relevant_years = float(match.group(2))
        return self.calculate_experience_score(relevant_years)

    def score_relevant_degree(self, resume_data):
        degree_text = self.llm_select_sections(resume_data, "Education")
        if not degree_text:
            return 0

        messages = [
            {"role": "system", "content": f"""
            Analyze the highest relevant degree.
            Exact matches: {', '.join(self.RELEVANT_DEGREES['exact_match'])}
            Related fields: {', '.join(self.RELEVANT_DEGREES['related'])}

            Return in this exact format:
            FIELD:Computer Science;LEVEL:masters
            Use 'none' if no degree is found.
            """},
            {"role": "user", "content": degree_text}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        match = re.search(r'FIELD:(.*?);LEVEL:(.*?)$', response.choices[0].message.content.strip())
        if not match:
            return 0

        field = match.group(1).strip()
        level = match.group(2).strip().lower()

        if field in self.RELEVANT_DEGREES["exact_match"]:
            base_score = 10 * self.RELEVANT_DEGREES["weights"]["exact_match"]
        elif field in self.RELEVANT_DEGREES["related"]:
            base_score = 10 * self.RELEVANT_DEGREES["weights"]["related"]
        else:
            base_score = 3

        level_multipliers = {
            "bachelors": 1.0,
            "masters": 1.2,
            "phd": 1.5,
            "none": 0.5
        }

        final_score = base_score * level_multipliers.get(level, 1.0)
        return round(final_score, 2)

    def calculate_final_score(self, scores: Dict[str, float]) -> float:
        base_score = sum(score * self.weights[category] for category, score in scores.items())
        return max(0, min(10, base_score))

    # Main scoring logic
    def evaluate_candidates(self, result):
        self.extract_target_criteria(self.job_listing)
        candidate_scores = {}
        for candidate, data in result.items():
            scores = {
                "Matching skills weight": self.score_matching_skills(data)["score"],
                "Missing skills weight": self.score_missing_skills(data)["score"],
                "Relevant job list weight": self.score_relevant_jobs(data),
                "Relevant degree list weight": self.score_relevant_degree(data),
                "Years of relevant experience weight": self.score_years_of_experience(data)
            }

            final_score = self.calculate_final_score(scores)
            scores["Final Score"] = round(final_score, 2)
            candidate_scores[candidate] = scores

        return dict(sorted(
            candidate_scores.items(),
            key=lambda x: x[1]["Final Score"],
            reverse=True
        ))







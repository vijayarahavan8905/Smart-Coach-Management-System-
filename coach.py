"""
Coach Lens - Module 7: Coach Feedback Advisor
=============================================
Translates quantitative kinematic metric gaps into clear, actionable coaching tips.

Architecture:
1. Rule-Based Core (Always Active & Zero Failure): Generates deterministic,
   biomechanically precise tips based on exact degree gaps and thresholds.
2. Optional LLM Rephraser (Gemini Free Tier): Rephrases measured feedback into
   natural, encouraging conversational coaching text when an API key is available.
3. Strict Safety Fallback: Automatically reverts to deterministic rules if the API
   is offline, quota is exceeded, or no key is provided.
"""

import os
from typing import Dict, List, Optional


class CoachingAdvisor:
    """
    Generates tailored coaching feedback from repetition evaluation results.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_client = None

        if self.api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                print(f"[COACH INFO] Gemini client initialization skipped ({exc}). Using rule-based fallback.")
                self.gemini_client = None

    def generate_rule_based_tips(self, eval_result: Dict) -> List[str]:
        """
        Deterministic, rule-based feedback derived strictly from measured physical gaps.
        """
        score = eval_result.get("overall_score", 0.0)
        gaps = eval_result.get("gaps", {})
        penalties = eval_result.get("penalties", {})
        worst = eval_result.get("worst_moment", {})

        tips: List[str] = []

        # 1. Depth Analysis
        depth_gap = gaps.get("depth_gap_degrees", 0.0)
        user_min_knee = gaps.get("user_min_knee", 0.0)
        if depth_gap > 0:
            tips.append(
                f"Squat Depth was shallow ({user_min_knee:.1f}° vs. target <= 90°). "
                f"You stopped approximately {depth_gap:.1f}° above parallel. "
                "Sink your hips deeper so your hip crease drops level with the top of your knees."
            )
        else:
            tips.append(
                f"Great squat depth! Your knees flexed to {user_min_knee:.1f}°, breaking parallel cleanly."
            )

        # 2. Torso Lean Analysis
        torso_gap = gaps.get("torso_gap_degrees", 0.0)
        user_peak_torso = gaps.get("user_peak_torso", 0.0)
        if torso_gap > 0:
            tips.append(
                f"Excessive forward chest lean detected ({user_peak_torso:.1f}° vs. recommended max 35°). "
                "Keep your chest proud, engage your core, and avoid pitching forward during the ascent."
            )
        else:
            tips.append(
                f"Solid spinal alignment! Your torso remained upright at {user_peak_torso:.1f}° throughout the rep."
            )

        # 3. Trajectory & Tempo (DTW)
        traj_penalty = penalties.get("trajectory_penalty", 0.0)
        if traj_penalty > 10.0:
            tips.append(
                "Pacing & Smoothness: Your movement path showed jitter or hesitation during the transition. "
                "Focus on a smooth, controlled 2-second descent and a steady drive upwards."
            )

        # 4. Summary Action Item
        if score >= 90:
            tips.insert(0, "🌟 Textbook Repetition! High biomechanical efficiency.")
        elif score >= 75:
            tips.insert(0, f"👍 Good rep (Score: {score}/100). Focus on: {worst.get('reason', 'polishing form')}.")
        else:
            tips.insert(0, f"⚠️ Form breakdown detected (Score: {score}/100). Primary issue: {worst.get('reason', 'check depth and posture')}.")

        return tips

    def get_coaching_feedback(self, eval_result: Dict, use_ai: bool = True) -> Dict:
        """
        Returns feedback dictionary with rule-based tips and optional AI rephrasing.
        """
        rule_tips = self.generate_rule_based_tips(eval_result)
        ai_summary = None

        if use_ai and self.gemini_client:
            try:
                gaps = eval_result.get("gaps", {})
                prompt = (
                    "You are 'Coach Lens', an encouraging, expert gym coach. "
                    "Review these exact measured workout metrics for one squat repetition:\n"
                    f"- Form Score: {eval_result.get('overall_score')}/100\n"
                    f"- Knee Depth Angle: {gaps.get('user_min_knee')} deg (Target: <= 90 deg, Gap: {gaps.get('depth_gap_degrees')} deg)\n"
                    f"- Torso Forward Lean: {gaps.get('user_peak_torso')} deg (Target max: 35 deg, Gap: {gaps.get('torso_gap_degrees')} deg)\n"
                    f"- Primary Defect: {eval_result.get('worst_moment', {}).get('reason')}\n\n"
                    "Instructions:\n"
                    "1. Write 2-3 concise, high-energy coaching sentences telling the lifter what they did well and how to fix their biggest mistake.\n"
                    "2. Only mention the exact numbers provided above; do not invent new measurements.\n"
                    "3. Keep the tone motivational and practical."
                )

                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                if response and response.text:
                    ai_summary = response.text.strip()
            except Exception as exc:
                print(f"[COACH INFO] Gemini AI rephrasing unavailable ({exc}). Using rule-based fallback.")
                ai_summary = None

        return {
            "score": eval_result.get("overall_score", 0.0),
            "grade": eval_result.get("grade", "N/A"),
            "rule_tips": rule_tips,
            "ai_coach_voice": ai_summary,
            "used_ai": ai_summary is not None,
        }

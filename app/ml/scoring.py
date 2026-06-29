"""
Scoring Module
================
Job Fit Score computation and score presentation utilities.

The Job Fit Score is the single most important metric in the system.
It's used to:
1. Rank job recommendations for candidates
2. Rank candidates for recruiters
3. Display match quality on dashboards
4. Filter recommendations (only show scores above threshold)

Score Formula:
    overall = 0.40 × cosine_similarity + 0.30 × skill_score
            + 0.15 × education_score + 0.15 × experience_score
"""


def get_score_label(score):
    """
    Convert a numerical score to a human-readable label.

    Args:
        score: Float between 0.0 and 1.0

    Returns:
        String label
    """
    if score >= 0.85:
        return 'Excellent Match'
    elif score >= 0.70:
        return 'Great Match'
    elif score >= 0.55:
        return 'Good Match'
    elif score >= 0.40:
        return 'Fair Match'
    elif score >= 0.25:
        return 'Partial Match'
    else:
        return 'Low Match'


def get_score_color(score):
    """
    Get the CSS color class for a score value.

    Args:
        score: Float between 0.0 and 1.0

    Returns:
        CSS color class string
    """
    if score >= 0.70:
        return 'success'
    elif score >= 0.40:
        return 'warning'
    else:
        return 'danger'


def format_score_percentage(score):
    """
    Format score as a percentage string.

    Args:
        score: Float between 0.0 and 1.0

    Returns:
        Formatted percentage string (e.g., "85.2%")
    """
    if score is None:
        return 'N/A'
    return f'{score * 100:.1f}%'


def compute_overall_score(cosine_sim, skill_score, education_score, experience_score):
    """
    Compute the final Job Fit Score from component scores.

    Args:
        cosine_sim: TF-IDF cosine similarity (0-1)
        skill_score: Skill matching score (0-1)
        education_score: Education matching score (0-1)
        experience_score: Experience matching score (0-1)

    Returns:
        Overall score clamped to [0.0, 1.0]
    """
    score = (
        0.40 * cosine_sim +
        0.30 * skill_score +
        0.15 * education_score +
        0.15 * experience_score
    )
    return min(1.0, max(0.0, round(score, 4)))

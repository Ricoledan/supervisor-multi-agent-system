class ResponseConfig:
    """Configuration settings for response formatting."""

    DETAILED_MODE = True
    INCLUDE_VISUALIZATIONS = True
    INCLUDE_CONFIDENCE = True
    MAX_SUMMARY_LENGTH = 200

    VISUALIZATION_TYPES = {
        "graph": True,
        "topics": True,
        "timeline": False,
        "histogram": False
    }

    STRUCTURE_ELEMENTS = {
        "summary": True,
        "sections": True,
        "recommendations": True,
        "related_topics": True,
        "references": True
    }
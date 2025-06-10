# backend/function_definitions.py
COLLECTION_FUNCTIONS = [
    {
        "name": "complete_data_collection",
        "description": "Call this function when ALL required user information has been successfully collected and validated",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User's full name"
                },
                "id": {
                    "type": "string",
                    "pattern": "^[0-9]{9}$",
                    "description": "9-digit Israeli ID number"
                },
                "gender": {
                    "type": "string",
                    "enum": ["male", "female", "other"],
                    "description": "User's gender"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 120,
                    "description": "User's age"
                },
                "hmo": {
                    "type": "string",
                    "enum": ["מכבי", "מאוחדת", "כללית"],
                    "description": "Health maintenance organization name"
                },
                "card": {
                    "type": "string",
                    "pattern": "^[0-9]{9}$",
                    "description": "9-digit HMO card number"
                },
                "tier": {
                    "type": "string",
                    "enum": ["זהב", "כסף", "ארד"],
                    "description": "Insurance membership tier"
                },
                "preferred_language": {
                    "type": "string",
                    "enum": ["he", "en"],
                    "description": "User's preferred language for communication"
                }
            },
            "required": ["name", "id", "gender", "age", "hmo", "card", "tier", "preferred_language"]
        }
    }
]
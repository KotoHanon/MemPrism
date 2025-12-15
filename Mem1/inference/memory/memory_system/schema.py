import json

class Schema:
    def __init__(self, max_slots = 10, max_tags = 5):
        self.QA_TASK_SLOT_SCHEMA = {
        "type": "object",
        "properties": {
            "slots": {
            "type": "array",
            "minItems": 1,
            "maxItems": max_slots,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                "stage": {"type": "string", "enum": [
                    "question_understanding", "information_retrieval",
                    "answer_generation", "answer_validation", "meta"
                ]},
                "topic": {"type": "string"},
                "summary": {"type": "string"},
                "attachments": {"type": ["object", "null"]},
                "tags": {
                    "type": "array",
                    "maxItems": max_tags,
                    "items": {"type": "string"}
                },
                },
                "required": ["stage", "topic", "summary", "attachments", "tags"],
            },
            }
        },
        "required": ["slots"],
        "additionalProperties": False,
        }

        self.FC_TASK_SLOT_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slots": {
            "type": "array",
            "minItems": 1,
            "maxItems": max_slots,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                "memory_type": {
                    "type": "string",
                    "enum": ["semantic", "episodic", "procedural"]
                },
                "stage": {
                    "type": "string",
                    "enum": [
                    "intent_constraints",
                    "tool_selection",
                    "argument_construction",
                    "tool_execution",
                    "result_integration",
                    "error_handling",
                    "meta"
                    ]
                },
                "topic": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 80,
                    "pattern": "^[a-z0-9]+(?: [a-z0-9]+){2,6}$"
                    # 3–7 words slug, lowercase, space-separated
                },
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 700
                },
                "attachments": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                    "constraints": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 20
                        }
                        },
                        "required": ["items"]
                    },
                    "tool_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30
                        }
                        },
                        "required": ["items"]
                    },
                    "arg_map": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30
                        }
                        },
                        "required": ["items"]
                    },
                    "observations": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30
                        }
                        },
                        "required": ["items"]
                    },
                    "failures": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30
                        }
                        },
                        "required": ["items"]
                    },
                    "recovery": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30
                        }
                        },
                        "required": ["steps"]
                    },
                    "checks": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30
                        }
                        },
                        "required": ["items"]
                    }
                    }
                },
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 6,
                    "items": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]*$"
                    }
                }
                },
                "required": ["memory_type", "stage", "topic", "summary", "attachments", "tags"]
            }
            }
        },
        "required": ["slots"]
        }
    
        self.EXPERIMENT_TASK_SLOT_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slots": {
            "type": "array",
            "minItems": 0, 
            "maxItems": max_slots,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                "stage": {
                    "type": "string",
                    "enum": [
                    "pre_analysis",
                    "code_plan",
                    "code_implement",
                    "code_judge",
                    "experiment_execute",
                    "experiment_analysis",
                    "meta"
                    ]
                },
                "topic": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 80,
                    "pattern": "^[a-z0-9]+(?: [a-z0-9]+){2,5}$"
                },
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000 
                },
                "attachments": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                    "notes": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 50
                        }
                        },
                        "required": ["items"]
                    },
                    "metrics": {
                        "type": "object",
                        "additionalProperties": {
                        "type": ["number", "string", "boolean", "null"]
                        }
                    },
                    "issues": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 50
                        }
                        },
                        "required": ["list"]
                    },
                    "actions": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                        "list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 50
                        }
                        },
                        "required": ["list"]
                    }
                    }
                },
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]*$"
                    }
                }
                },
                "required": ["stage", "topic", "summary", "attachments", "tags"]
            }
            }
        },
        "required": ["slots"]
        }
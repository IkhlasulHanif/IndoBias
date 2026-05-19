SCHEMA_REFERENCE = {
    "overview": {
        "dimensions_file": "data/dimensions.py",
        "groups_dir": "data/groups",
        "purpose": [
            "dimensions.py defines stereotype dimensions and label pairs",
            "groups/*.csv defines the target demographic groups",
        ],
    },
    "dimensions_python": {
        "global_name": "DIMENSION_CATALOG",
        "top_level_fields": {
            "name": "string",
            "version": "string",
            "description": "string",
            "dimensions": "list[DimensionDefinition]",
        },
        "dimension_definition": {
            "required": ["dimension", "name", "pairs"],
            "fields": {
                "dimension": {
                    "type": "string",
                    "allowed": ["wgi", "onet", "qs_rankings", "sp"],
                },
                "name": "string",
                "description": "string",
                "pairs": "list[StereotypePair]",
            },
        },
        "stereotype_pair": {
            "required": ["pair_id", "positive_label", "negative_label"],
            "fields": {
                "pair_id": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*$",
                },
                "parent_category": "string",
                "positive_label": "string",
                "negative_label": "string",
                "description": "string",
            },
        },
    },
    "groups_csv": {
        "file_pattern": ["{demography}.csv", "{demography}_{region}.csv"],
        "inferred_from_filename": {
            "ethnicity": "ethnicity",
            "ethnicity_indonesia": "ethnicity",
            "gender": "gender",
            "lembaga": "lembaga",
            "media": "media",
            "name": "name",
            "nationality": "nationality",
            "organisasi_bisnis": "organisasi_bisnis",
            "political_party": "political_party",
            "race": "race",
            "religion": "religion",
            "university": "university",
        },
        "special_country_defaults": {
            "ethnicity_indonesia": "Indonesia",
        },
        "row_fields": {
            "required": ["group_id", "group_name"],
            "fields": {
                "group_id": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*$",
                },
                "group_name": "string",
                "demography": {
                    "type": "derived",
                    "source": "filename",
                },
                "country": {
                    "type": "optional",
                    "source": "filename default or empty",
                },
            },
        },
    },
}

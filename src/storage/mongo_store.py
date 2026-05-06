from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.logging import get_logger

try:
    from pymongo import MongoClient
except ModuleNotFoundError:  # pragma: no cover
    MongoClient = None  # type: ignore

mongo_logger = get_logger("mongo_store")
ANALYTICS_ROUTES = {
    "/analyze-case-pdf",
    "/jobs/analyze-case-pdf",
    "/jobs/analyze-case-batch/item",
}


class MongoRunStore:
    def __init__(self) -> None:
        self.uri = (
            os.getenv("MONGODB_URI", "").strip()
            or os.getenv("MONGO_URI", "").strip()
        )
        self.db_name = os.getenv("MONGODB_DB", "violence_judicial_ai").strip() or "violence_judicial_ai"
        self.collection_name = os.getenv("MONGODB_COLLECTION", "processing_runs").strip() or "processing_runs"
        self.auth_collection_name = (
            os.getenv("MONGODB_AUTH_COLLECTION", "auth_users").strip() or "auth_users"
        )
        self.benchmark_collection_name = (
            os.getenv("MONGODB_BENCHMARK_COLLECTION", "benchmark_runs").strip() or "benchmark_runs"
        )
        self.anonymization_issue_collection_name = (
            os.getenv("MONGODB_ANONYMIZATION_ISSUE_COLLECTION", "anonymization_error_cases").strip()
            or "anonymization_error_cases"
        )
        self.legal_benchmark_collection_name = (
            os.getenv("MONGODB_LEGAL_BENCHMARK_COLLECTION", "measures_legal_corpus_benchmark").strip()
            or "measures_legal_corpus_benchmark"
        )
        self.last_error: str | None = None

    def enabled(self) -> bool:
        return bool(self.uri and MongoClient is not None)

    def _clear_error(self) -> None:
        self.last_error = None

    def _set_error(self, message: str) -> None:
        self.last_error = message
        mongo_logger.error(message)

    def status_snapshot(self) -> dict[str, Any]:
        if not self.uri:
            return {
                "enabled": False,
                "status": "disabled",
                "message": "MONGODB_URI no está configurada.",
                "db": self.db_name,
                "collection": self.collection_name,
                "auth_collection": self.auth_collection_name,
                "benchmark_collection": self.benchmark_collection_name,
                "anonymization_issue_collection": self.anonymization_issue_collection_name,
                "legal_benchmark_collection": self.legal_benchmark_collection_name,
            }
        if MongoClient is None:
            return {
                "enabled": False,
                "status": "driver_missing",
                "message": "pymongo no está instalado en el entorno actual.",
                "db": self.db_name,
                "collection": self.collection_name,
                "auth_collection": self.auth_collection_name,
                "benchmark_collection": self.benchmark_collection_name,
                "anonymization_issue_collection": self.anonymization_issue_collection_name,
                "legal_benchmark_collection": self.legal_benchmark_collection_name,
            }
        return {
            "enabled": True,
            "status": "ok" if not self.last_error else "error",
            "message": self.last_error or "MongoDB listo.",
            "db": self.db_name,
            "collection": self.collection_name,
            "auth_collection": self.auth_collection_name,
            "benchmark_collection": self.benchmark_collection_name,
            "anonymization_issue_collection": self.anonymization_issue_collection_name,
            "legal_benchmark_collection": self.legal_benchmark_collection_name,
        }

    def _create_client(self, context: str) -> Any | None:
        if not self.enabled():
            return None
        try:
            return MongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
        except Exception as exc:
            self._set_error(f"Mongo {context} client init error: {exc}")
            return None

    @staticmethod
    def _close_client(client: Any | None) -> None:
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass

    def ensure_indexes(self) -> dict[str, Any]:
        if not self.enabled():
            return {"enabled": False, "created": []}
        client = self._create_client("bootstrap")
        if client is None:
            return {"enabled": True, "created": [], "error": self.last_error}
        try:
            client.admin.command("ping")
            self._clear_error()
            collection = client[self.db_name][self.collection_name]
            auth_collection = client[self.db_name][self.auth_collection_name]
            benchmark_collection = client[self.db_name][self.benchmark_collection_name]
            anonymization_issue_collection = client[self.db_name][self.anonymization_issue_collection_name]
            legal_benchmark_collection = client[self.db_name][self.legal_benchmark_collection_name]
            created = [
                collection.create_index("created_at", name="idx_created_at_desc"),
                collection.create_index("route", name="idx_route"),
                collection.create_index("filename", name="idx_filename"),
                collection.create_index("case_id", name="idx_case_id"),
                collection.create_index(
                    "classification.selected_template_id",
                    name="idx_selected_template_id",
                ),
                collection.create_index(
                    "classification.selected_template_name",
                    name="idx_selected_template_name",
                ),
                collection.create_index(
                    "classification.risk_level",
                    name="idx_risk_level",
                ),
                collection.create_index("runtime.provider", name="idx_runtime_provider"),
                collection.create_index("job_id", name="idx_job_id"),
                auth_collection.create_index("username", unique=True, name="idx_auth_username_unique"),
                auth_collection.create_index("role", name="idx_auth_role"),
                auth_collection.create_index("active", name="idx_auth_active"),
                benchmark_collection.create_index("created_at", name="idx_benchmark_created_at_desc"),
                benchmark_collection.create_index("label", name="idx_benchmark_label"),
                benchmark_collection.create_index("benchmark_id", unique=True, name="idx_benchmark_id_unique"),
                anonymization_issue_collection.create_index("created_at", name="idx_anonymization_issue_created_at_desc"),
                anonymization_issue_collection.create_index("case_id", name="idx_anonymization_issue_case_id"),
                anonymization_issue_collection.create_index("filename", name="idx_anonymization_issue_filename"),
                anonymization_issue_collection.create_index("job_id", name="idx_anonymization_issue_job_id"),
                anonymization_issue_collection.create_index("route", name="idx_anonymization_issue_route"),
                legal_benchmark_collection.create_index("created_at", name="idx_legal_benchmark_created_at_desc"),
                legal_benchmark_collection.create_index("label", name="idx_legal_benchmark_label"),
                legal_benchmark_collection.create_index("benchmark_id", unique=True, name="idx_legal_benchmark_id_unique"),
            ]
            return {"enabled": True, "created": created}
        except Exception as exc:
            self._set_error(f"Mongo bootstrap error: {exc}")
            return {"enabled": True, "created": [], "error": str(exc)}
        finally:
            self._close_client(client)

    def upsert_auth_user(
        self,
        *,
        username: str,
        password_hash: str,
        role: str = "admin",
        active: bool = True,
    ) -> bool:
        if not self.enabled():
            return False
        client = self._create_client("upsert_auth_user")
        if client is None:
            return False
        try:
            client.admin.command("ping")
            self._clear_error()
            now = datetime.now(timezone.utc).isoformat()
            result = client[self.db_name][self.auth_collection_name].update_one(
                {"username": username},
                {
                    "$set": {
                        "username": username,
                        "password_hash": password_hash,
                        "role": role,
                        "active": active,
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            return bool(result.acknowledged)
        except Exception as exc:
            self._set_error(f"Mongo upsert_auth_user error: {exc}")
            return False
        finally:
            self._close_client(client)

    def fetch_auth_user(self, username: str) -> dict[str, Any] | None:
        if not self.enabled():
            return None
        client = self._create_client("fetch_auth_user")
        if client is None:
            return None
        try:
            client.admin.command("ping")
            self._clear_error()
            return client[self.db_name][self.auth_collection_name].find_one(
                {"username": username},
                {"_id": 0},
            )
        except Exception as exc:
            self._set_error(f"Mongo fetch_auth_user error: {exc}")
            return None
        finally:
            self._close_client(client)

    def create_auth_user(
        self,
        *,
        username: str,
        password_hash: str,
        role: str = "operator",
        active: bool = True,
    ) -> tuple[bool, str | None]:
        if not self.enabled():
            return False, "mongo_disabled"
        client = self._create_client("create_auth_user")
        if client is None:
            return False, "unavailable"
        try:
            client.admin.command("ping")
            self._clear_error()
            collection = client[self.db_name][self.auth_collection_name]
            existing = collection.find_one({"username": username}, {"_id": 1})
            if existing is not None:
                return False, "duplicate"
            now = datetime.now(timezone.utc).isoformat()
            result = collection.insert_one(
                {
                    "username": username,
                    "password_hash": password_hash,
                    "role": role,
                    "active": active,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            return bool(result.acknowledged), None
        except Exception as exc:
            self._set_error(f"Mongo create_auth_user error: {exc}")
            return False, "error"
        finally:
            self._close_client(client)

    def save_run(self, payload: dict[str, Any]) -> str | None:
        if not self.enabled():
            return None
        client = self._create_client("save_run")
        if client is None:
            return None
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.collection_name].insert_one(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **payload,
                }
            )
            return str(result.inserted_id)
        except Exception as exc:
            self._set_error(f"Mongo save_run error: {exc}")
            return None
        finally:
            self._close_client(client)

    def save_anonymization_issue(self, payload: dict[str, Any]) -> str | None:
        if not self.enabled():
            return None
        client = self._create_client("save_anonymization_issue")
        if client is None:
            return None
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.anonymization_issue_collection_name].insert_one(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **payload,
                }
            )
            return str(result.inserted_id)
        except Exception as exc:
            self._set_error(f"Mongo save_anonymization_issue error: {exc}")
            return None
        finally:
            self._close_client(client)

    @staticmethod
    def _analytics_query() -> dict[str, Any]:
        return {"route": {"$in": sorted(ANALYTICS_ROUTES)}}

    def fetch_recent_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        client = self._create_client("fetch_recent_runs")
        if client is None:
            return []
        try:
            client.admin.command("ping")
            self._clear_error()
            cursor = (
                client[self.db_name][self.collection_name]
                .find(self._analytics_query(), {"_id": 0})
                .sort("created_at", -1)
                .limit(max(1, limit))
            )
            return list(cursor)
        except Exception as exc:
            self._set_error(f"Mongo fetch_recent_runs error: {exc}")
            return []
        finally:
            self._close_client(client)

    def search_runs(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        needle = (query or "").strip()
        if not needle:
            return []
        client = self._create_client("search_runs")
        if client is None:
            return []
        try:
            client.admin.command("ping")
            self._clear_error()
            escaped = {"$regex": needle, "$options": "i"}
            mongo_query: dict[str, Any] = {
                "$or": [
                    {"case_id": escaped},
                    {"filename": escaped},
                    {"classification.selected_template_name": escaped},
                    {"classification.selected_template_id": escaped},
                    {"classification.risk_level": escaped},
                    {"pipeline_result.extracted_case.anonymized_text": escaped},
                ]
            }
            cursor = (
                client[self.db_name][self.collection_name]
                .find(mongo_query, {"_id": 0})
                .sort("created_at", -1)
                .limit(max(1, limit))
            )
            return list(cursor)
        except Exception as exc:
            self._set_error(f"Mongo search_runs error: {exc}")
            return []
        finally:
            self._close_client(client)

    def mark_run_validated(
        self,
        *,
        created_at: str,
        case_id: str | None = None,
        filename: str | None = None,
        route: str | None = None,
        valid: bool = True,
    ) -> bool:
        if not self.enabled():
            return False
        client = self._create_client("mark_run_validated")
        if client is None:
            return False
        try:
            client.admin.command("ping")
            self._clear_error()
            query: dict[str, Any] = {"created_at": created_at}
            if case_id is not None:
                query["case_id"] = case_id
            if filename is not None:
                query["filename"] = filename
            if route is not None:
                query["route"] = route
            result = client[self.db_name][self.collection_name].update_one(
                query,
                {
                    "$set": {
                        "review.validated": valid,
                        "review.validated_at": datetime.now(timezone.utc).isoformat(),
                        "review.decision": "correct" if valid else "rejected",
                    }
                },
            )
            return bool(result.modified_count or result.matched_count)
        except Exception as exc:
            self._set_error(f"Mongo mark_run_validated error: {exc}")
            return False
        finally:
            self._close_client(client)

    def delete_runs(self, query: dict[str, Any] | None = None) -> int:
        if not self.enabled():
            return 0
        client = self._create_client("delete_runs")
        if client is None:
            return 0
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.collection_name].delete_many(query or {})
            return int(result.deleted_count)
        except Exception as exc:
            self._set_error(f"Mongo delete_runs error: {exc}")
            raise
        finally:
            self._close_client(client)

    def delete_one_run(self, query: dict[str, Any]) -> int:
        if not self.enabled():
            return 0
        client = self._create_client("delete_one_run")
        if client is None:
            return 0
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.collection_name].delete_one(query)
            return int(result.deleted_count)
        except Exception as exc:
            self._set_error(f"Mongo delete_one_run error: {exc}")
            raise
        finally:
            self._close_client(client)

    def delete_anonymization_issues(self, query: dict[str, Any]) -> int:
        if not self.enabled():
            return 0
        client = self._create_client("delete_anonymization_issues")
        if client is None:
            return 0
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.anonymization_issue_collection_name].delete_many(query)
            return int(result.deleted_count)
        except Exception as exc:
            self._set_error(f"Mongo delete_anonymization_issues error: {exc}")
            raise
        finally:
            self._close_client(client)

    def save_benchmark_run(self, payload: dict[str, Any]) -> str | None:
        if not self.enabled():
            return None
        client = self._create_client("save_benchmark_run")
        if client is None:
            return None
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.benchmark_collection_name].insert_one(payload)
            return str(result.inserted_id)
        except Exception as exc:
            self._set_error(f"Mongo save_benchmark_run error: {exc}")
            return None
        finally:
            self._close_client(client)

    def save_legal_benchmark_run(self, payload: dict[str, Any]) -> str | None:
        if not self.enabled():
            return None
        client = self._create_client("save_legal_benchmark_run")
        if client is None:
            return None
        try:
            client.admin.command("ping")
            self._clear_error()
            result = client[self.db_name][self.legal_benchmark_collection_name].insert_one(payload)
            return str(result.inserted_id)
        except Exception as exc:
            self._set_error(f"Mongo save_legal_benchmark_run error: {exc}")
            return None
        finally:
            self._close_client(client)

    def fetch_legal_benchmark_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        client = self._create_client("fetch_legal_benchmark_runs")
        if client is None:
            return []
        try:
            client.admin.command("ping")
            self._clear_error()
            cursor = (
                client[self.db_name][self.legal_benchmark_collection_name]
                .find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(max(1, limit))
            )
            return list(cursor)
        except Exception as exc:
            self._set_error(f"Mongo fetch_legal_benchmark_runs error: {exc}")
            return []
        finally:
            self._close_client(client)

    def fetch_benchmark_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        client = self._create_client("fetch_benchmark_runs")
        if client is None:
            return []
        try:
            client.admin.command("ping")
            self._clear_error()
            cursor = (
                client[self.db_name][self.benchmark_collection_name]
                .find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(max(1, limit))
            )
            return list(cursor)
        except Exception as exc:
            self._set_error(f"Mongo fetch_benchmark_runs error: {exc}")
            return []
        finally:
            self._close_client(client)

    def metrics_summary(self, limit: int = 300) -> dict[str, Any]:
        if not self.enabled():
            return {
                "enabled": False,
                "mongo": self.status_snapshot(),
                "total_runs": 0,
                "completed_runs": 0,
                "failed_runs": 0,
                "templates": [],
                "risk_levels": [],
                "providers": [],
                "alerts": [],
                "daily_runs": [],
            }
        client = self._create_client("metrics_summary")
        if client is None:
            return {
                "enabled": False,
                "mongo": self.status_snapshot(),
                "total_runs": 0,
                "completed_runs": 0,
                "failed_runs": 0,
                "templates": [],
                "risk_levels": [],
                "providers": [],
                "alerts": [],
                "daily_runs": [],
            }
        try:
            client.admin.command("ping")
            self._clear_error()
            docs = list(
                client[self.db_name][self.collection_name]
                .find(self._analytics_query(), {"_id": 0})
                .sort("created_at", -1)
                .limit(max(1, limit))
            )
        except Exception as exc:
            self._set_error(f"Mongo metrics_summary error: {exc}")
            return {
                "enabled": False,
                "mongo": self.status_snapshot(),
                "total_runs": 0,
                "completed_runs": 0,
                "failed_runs": 0,
                "templates": [],
                "risk_levels": [],
                "providers": [],
                "alerts": [],
                "daily_runs": [],
            }
        finally:
            self._close_client(client)

        template_counts: Counter[str] = Counter()
        risk_counts: Counter[str] = Counter()
        provider_counts: Counter[str] = Counter()
        alert_counts: Counter[str] = Counter()
        daily_counts: Counter[str] = Counter()
        completed_runs = 0
        failed_runs = 0

        for doc in docs:
            created_at = str(doc.get("created_at", ""))[:10]
            if created_at:
                daily_counts[created_at] += 1

            error = doc.get("error")
            if error:
                failed_runs += 1

            runtime = doc.get("runtime") or {}
            provider = runtime.get("provider")
            if provider:
                provider_counts[str(provider)] += 1

            classification = doc.get("classification") or {}
            if classification:
                completed_runs += 1
                template = classification.get("selected_template_name") or classification.get("selected_template_id")
                if template:
                    template_counts[str(template)] += 1
                risk = classification.get("risk_level")
                if risk:
                    risk_counts[str(risk)] += 1
                for alert in classification.get("alerts") or []:
                    if str(alert).strip():
                        alert_counts[str(alert).strip()] += 1

        return {
            "enabled": True,
            "mongo": self.status_snapshot(),
            "total_runs": len(docs),
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "templates": [{"name": key, "count": value} for key, value in template_counts.most_common(8)],
            "risk_levels": [{"name": key, "count": value} for key, value in risk_counts.most_common()],
            "providers": [{"name": key, "count": value} for key, value in provider_counts.most_common()],
            "alerts": [{"name": key, "count": value} for key, value in alert_counts.most_common(8)],
            "daily_runs": [{"date": key, "count": value} for key, value in sorted(daily_counts.items())],
        }


def build_run_payload(
    *,
    route: str,
    job_id: str | None = None,
    case_id: str | None = None,
    filename: str | None = None,
    ocr_backend: str | None = None,
    tesseract_lang: str | None = None,
    ollama_model: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    anonymization_audit: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    pipeline_result: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    generated_draft: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "route": route,
        "job_id": job_id,
        "case_id": case_id,
        "filename": filename,
        "runtime": {
            "ocr_backend": ocr_backend,
            "tesseract_lang": tesseract_lang,
            "ollama_model": ollama_model,
            "provider": provider,
            "model": model,
        },
        "anonymization_audit": anonymization_audit,
        "timings": timings,
        "pipeline_result": pipeline_result,
        "classification": classification,
        "generated_draft": generated_draft,
        "error": error,
    }


def build_anonymization_issue_payload(
    *,
    route: str,
    job_id: str | None = None,
    case_id: str | None = None,
    filename: str | None = None,
    ocr_backend: str | None = None,
    tesseract_lang: str | None = None,
    ollama_model: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    anonymization_audit: dict[str, Any] | None = None,
    anonymization_warning: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "route": route,
        "job_id": job_id,
        "case_id": case_id,
        "filename": filename,
        "runtime": {
            "ocr_backend": ocr_backend,
            "tesseract_lang": tesseract_lang,
            "ollama_model": ollama_model,
            "provider": provider,
            "model": model,
        },
        "anonymization_audit": anonymization_audit,
        "anonymization_warning": anonymization_warning,
        "error": error,
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, run_migrations
from .models.models import Base
from .api import (
    documents, extraction, datasets, training, assistant, assistants,
    evaluations, pretraining,
)

Base.metadata.create_all(bind=engine)
run_migrations()

app = FastAPI(title="CDSS 医学知识蒸馏平台", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api")
app.include_router(extraction.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")
app.include_router(assistants.router, prefix="/api")
app.include_router(evaluations.router, prefix="/api")
app.include_router(pretraining.router, prefix="/api")


@app.get("/api/stats")
def global_stats():
    from .database import SessionLocal
    from .models.models import Document, ExtractionJob, DiagnosticInstance, Dataset, TrainingExperiment
    db = SessionLocal()
    try:
        return {
            "documents": db.query(Document).count(),
            "extraction_jobs": db.query(ExtractionJob).count(),
            "diagnostic_instances": db.query(DiagnosticInstance).count(),
            "datasets": db.query(Dataset).count(),
            "experiments": db.query(TrainingExperiment).count(),
        }
    finally:
        db.close()

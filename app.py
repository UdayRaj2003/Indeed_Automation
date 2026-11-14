"""Indeed Applied Jobs History web UI (mirrors Linkedin_Test/app.py via FastAPI)."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Indeed Applied Jobs History")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "all excels" / "all_applied_applications_history.csv"
TEMPLATE_PATH = REPO_ROOT / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not TEMPLATE_PATH.is_file():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/applied-jobs")
def get_applied_jobs():
    if not CSV_PATH.is_file():
        return JSONResponse({"error": "No applications history found"}, status_code=404)
    try:
        jobs = []
        with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                jobs.append(
                    {
                        "Job_ID": row.get("Job ID", ""),
                        "Title": row.get("Title", ""),
                        "Company": row.get("Company", ""),
                        "HR_Name": row.get("HR Name", "Unknown"),
                        "HR_Link": row.get("HR Link", "Unknown"),
                        "Job_Link": row.get("Job Link", ""),
                        "External_Job_link": row.get("External Job link", ""),
                        "Date_Applied": row.get("Date Applied", ""),
                    }
                )
        return JSONResponse(jobs)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.put("/applied-jobs/{job_id}")
def update_applied_date(job_id: str):
    if not CSV_PATH.is_file():
        return JSONResponse({"error": f"CSV file not found at {CSV_PATH}"}, status_code=404)
    try:
        data = []
        found = False
        with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                if row.get("Job ID") == job_id:
                    row["Date Applied"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    found = True
                data.append(row)

        if not found:
            return JSONResponse({"error": f"Job ID {job_id} not found"}, status_code=404)

        with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return JSONResponse({"message": "Date Applied updated successfully"}, status_code=200)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5001)

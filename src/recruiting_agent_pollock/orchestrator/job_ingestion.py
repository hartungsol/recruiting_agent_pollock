"""
Job description ingestion module.

Provides functionality to ingest job descriptions from raw text or files,
normalize them using LLM, and persist to the database.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.schemas import JobDescription

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _read_docx(file_path: Path) -> str:
    """
    Read text content from a .docx file.
    
    Args:
        file_path: Path to the .docx file.
        
    Returns:
        Extracted text content.
        
    Raises:
        ImportError: If python-docx is not installed.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required to read .docx files. "
            "Install it with: pip install python-docx"
        )
    
    doc = Document(str(file_path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


class JobIngestionService:
    """
    Service for ingesting and normalizing job descriptions.
    
    Uses LLM to parse unstructured job description text into the
    structured JobDescription schema.
    """

    PARSING_PROMPT = """You are a job description parser. Extract structured information from the following job posting.

Job Description Text:
\"\"\"
{raw_text}
\"\"\"

Extract the following fields and return as JSON:
{{
    "title": "<job title>",
    "company_name": "<company name if mentioned, else empty string>",
    "min_experience_years": <minimum years required as number or null>,
    "max_violations_3y": <max violations allowed in 3 years as number or null>,
    "home_time": "<home time policy like 'weekly', 'bi-weekly', 'OTR' or null>",
    "equipment": ["<equipment/endorsement1>", "<equipment2>", ...],
    "knockout_rules": ["<disqualifying rule1>", "<rule2>", ...],
    "required_skills": ["<required skill1>", "<skill2>", ...],
    "preferred_skills": ["<preferred skill1>", "<skill2>", ...],
    "preferred_criteria": {{
        "experience": ["<preferred experience1>", ...],
        "certifications": ["<cert1>", ...],
        "other": ["<other preferences>", ...]
    }},
    "benefits": ["<benefit1>", "<benefit2>", ...],
    "salary_range": "<salary range if mentioned, else empty string>",
    "location": "<location if mentioned, else empty string>",
    "job_type": "<full-time|part-time|contract>"
}}

Rules:
- Extract knockout rules as clear disqualifying criteria (e.g., "DUI in last 5 years", "No valid CDL")
- Equipment includes endorsements like "tanker", "hazmat", "doubles/triples"
- If a field is not mentioned, use null for numbers or empty string/list for text/arrays
- Be precise with experience years - extract the minimum required
- For violations, this typically refers to moving violations or accidents

Only return valid JSON, no other text."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the job ingestion service.
        
        Args:
            llm_client: LLM client for parsing. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()

    async def ingest_from_text(
        self,
        raw_text: str,
        job_id: str | None = None,
    ) -> JobDescription:
        """
        Ingest a job description from raw text.
        
        Args:
            raw_text: Raw job description text.
            job_id: Optional job ID. Generated if not provided.
            
        Returns:
            Normalized JobDescription.
        """
        if not job_id:
            job_id = f"job-{uuid4().hex[:8]}"
        
        logger.info(f"Ingesting job description: {job_id}")
        
        # Try LLM-based parsing
        try:
            parsed = await self._parse_with_llm(raw_text)
        except OllamaError as e:
            logger.warning(f"LLM parsing failed, using fallback: {e}")
            parsed = self._fallback_parse(raw_text)
        
        # Build JobDescription from parsed data
        return JobDescription(
            job_id=job_id,
            title=parsed.get("title", "Unknown Position"),
            company_name=parsed.get("company_name", ""),
            raw_text=raw_text,
            min_experience_years=parsed.get("min_experience_years"),
            max_violations_3y=parsed.get("max_violations_3y"),
            home_time=parsed.get("home_time"),
            equipment=parsed.get("equipment", []),
            knockout_rules=parsed.get("knockout_rules", []),
            required_skills=parsed.get("required_skills", []),
            preferred_skills=parsed.get("preferred_skills", []),
            preferred_criteria=parsed.get("preferred_criteria", {}),
            benefits=parsed.get("benefits", []),
            salary_range=parsed.get("salary_range", ""),
            location=parsed.get("location", ""),
            job_type=parsed.get("job_type", "full-time"),
        )

    async def ingest_from_file(
        self,
        file_path: str | Path,
        job_id: str | None = None,
    ) -> JobDescription:
        """
        Ingest a job description from a text or docx file.
        
        Args:
            file_path: Path to the job description file (.txt, .docx, etc.).
            job_id: Optional job ID. Derived from filename if not provided.
            
        Returns:
            Normalized JobDescription.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is empty.
            ImportError: If reading .docx but python-docx not installed.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Job description file not found: {path}")
        
        # Read file based on extension
        suffix = path.suffix.lower()
        if suffix == ".docx":
            raw_text = _read_docx(path).strip()
        else:
            # Default to text file reading (.txt, .md, etc.)
            raw_text = path.read_text(encoding="utf-8").strip()
        
        if not raw_text:
            raise ValueError(f"Job description file is empty: {path}")
        
        # Use filename as job_id if not provided
        if not job_id:
            job_id = path.stem
        
        logger.info(f"Ingesting job from file: {path}")
        return await self.ingest_from_text(raw_text, job_id)

    async def _parse_with_llm(self, raw_text: str) -> dict:
        """
        Parse job description using LLM.
        
        Args:
            raw_text: Raw job description text.
            
        Returns:
            Parsed job data as dictionary.
            
        Raises:
            OllamaError: If LLM parsing fails or returns empty/invalid data.
        """
        prompt = self.PARSING_PROMPT.format(raw_text=raw_text[:8000])  # Truncate if too long
        
        response = await self._llm_client.chat_with_json(
            messages=[Message(role="user", content=prompt)],
        )
        
        if not response:
            raise OllamaError("Empty response from LLM")
        
        # Check if we got a meaningful response (at least a title)
        if not response.get("title"):
            logger.warning("LLM response missing required 'title' field")
            raise OllamaError("LLM response missing required fields")
        
        return response

    def _fallback_parse(self, raw_text: str) -> dict:
        """
        Simple regex-based fallback parser when LLM fails.
        
        Args:
            raw_text: Raw job description text.
            
        Returns:
            Parsed job data as dictionary.
        """
        result = {
            "title": "",
            "company_name": "",
            "min_experience_years": None,
            "max_violations_3y": None,
            "home_time": None,
            "equipment": [],
            "knockout_rules": [],
            "required_skills": [],
            "preferred_skills": [],
            "preferred_criteria": {},
            "benefits": [],
            "salary_range": "",
            "location": "",
            "job_type": "full-time",
        }
        
        text_lower = raw_text.lower()
        
        # Try to extract title from first line
        lines = raw_text.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100:  # Likely a title
                result["title"] = first_line
        
        # Extract experience requirements
        exp_patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
            r"minimum\s*(?:of\s*)?(\d+)\s*years?",
            r"at\s*least\s*(\d+)\s*years?",
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result["min_experience_years"] = float(match.group(1))
                break
        
        # Extract equipment/endorsements
        equipment_keywords = ["tanker", "hazmat", "doubles", "triples", "cdl-a", "cdl-b", "twic"]
        for keyword in equipment_keywords:
            if keyword in text_lower:
                result["equipment"].append(keyword.upper())
        
        # Extract home time
        home_patterns = [
            (r"home\s*(?:every\s*)?week", "weekly"),
            (r"home\s*(?:every\s*)?2\s*weeks?", "bi-weekly"),
            (r"bi-?weekly\s*home", "bi-weekly"),
            (r"regional", "regional"),
            (r"otr|over\s*the\s*road", "OTR"),
            (r"local", "local"),
        ]
        for pattern, home_time in home_patterns:
            if re.search(pattern, text_lower):
                result["home_time"] = home_time
                break
        
        # Extract salary if mentioned
        salary_pattern = r"\$[\d,]+(?:\s*-\s*\$[\d,]+)?(?:\s*(?:per|\/)\s*(?:year|hour|week|mile))?"
        salary_match = re.search(salary_pattern, raw_text, re.IGNORECASE)
        if salary_match:
            result["salary_range"] = salary_match.group(0)
        
        # Common knockout rules to look for
        knockout_keywords = [
            (r"no\s*dui|dui\s*(?:within|in\s*(?:the\s*)?last)\s*\d+\s*years?", "No DUI violations"),
            (r"clean\s*(?:driving\s*)?record", "Clean driving record required"),
            (r"no\s*(?:major\s*)?accidents?\s*(?:in|within)", "No recent accidents"),
            (r"valid\s*cdl", "Valid CDL required"),
            (r"must\s*pass\s*(?:drug|dot)\s*(?:test|screen)", "Must pass drug screening"),
        ]
        for pattern, rule in knockout_keywords:
            if re.search(pattern, text_lower):
                result["knockout_rules"].append(rule)
        
        # Extract benefits
        benefit_keywords = ["health insurance", "401k", "dental", "vision", "pto", "paid time off", 
                          "vacation", "bonus", "sign-on bonus", "medical"]
        for benefit in benefit_keywords:
            if benefit in text_lower:
                result["benefits"].append(benefit.title())
        
        return result


# Convenience function for simple usage
async def ingest_job_description(
    text_or_path: str | Path,
    job_id: str | None = None,
    llm_client: LLMClient | None = None,
) -> JobDescription:
    """
    Convenience function to ingest a job description.
    
    Args:
        text_or_path: Either raw text or path to a file.
        job_id: Optional job ID.
        llm_client: Optional LLM client.
        
    Returns:
        Normalized JobDescription.
    """
    service = JobIngestionService(llm_client=llm_client)
    
    # Check if it's a file path
    path = Path(text_or_path) if isinstance(text_or_path, str) else text_or_path
    if path.exists() and path.is_file():
        return await service.ingest_from_file(path, job_id)
    
    # Treat as raw text
    return await service.ingest_from_text(str(text_or_path), job_id)

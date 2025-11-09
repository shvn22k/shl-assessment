import os
import sys
import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'src'))

from hybrid_rag import hybrid_chat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommendation API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: list[AssessmentResponse]

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    start_time = time.time()
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Got query: {query[:100]}...")
    
    try:
        # get recommendations
        results = hybrid_chat(query, top_k=10, return_results=True, quiet=True)
        
        if not results:
            logger.warning(f"No results for: {query[:100]}")
            return RecommendationResponse(recommended_assessments=[])
        
        recommended_assessments = []
        for result in results[:10]:
            # handle test types
            test_types = result.get('test_types', [])
            if isinstance(test_types, str):
                test_types = [t.strip() for t in test_types.split(',') if t.strip()]
            elif not isinstance(test_types, list):
                test_types = []
            
            if not test_types:
                test_types = ["Unknown"]
            
            # parse duration
            duration = result.get('duration', '')
            try:
                if isinstance(duration, str):
                    duration = int(duration) if duration.isdigit() else 0
                else:
                    duration = int(duration) if duration else 0
            except:
                duration = 0
            
            assessment = AssessmentResponse(
                url=result.get('link') or result.get('url', ''),
                name=result.get('name', ''),
                adaptive_support=result.get('adaptive_support', 'No'),
                description=result.get('description', '')[:500] if result.get('description') else '',
                duration=duration,
                remote_support=result.get('remote_support', 'No'),
                test_type=test_types
            )
            recommended_assessments.append(assessment)
        
        elapsed = time.time() - start_time
        logger.info(f"Processed in {elapsed:.2f}s, returned {len(recommended_assessments)} results")
        
        return RecommendationResponse(recommended_assessments=recommended_assessments)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


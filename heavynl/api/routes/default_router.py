from fastapi import APIRouter
from fastapi.responses import RedirectResponse


defaultrouter = APIRouter()


@defaultrouter.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")

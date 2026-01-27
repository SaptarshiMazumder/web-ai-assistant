from typing import Optional

from domain.entities import UserRecord
from domain.repositories import UserRepository


class UserService:
    def __init__(self, user_repo: UserRepository) -> None:
        self._user_repo = user_repo

    def upsert_user_from_claims(
        self,
        *,
        subject: str,
        email: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> UserRecord:
        return self._user_repo.upsert_user_from_claims(
            subject=subject,
            email=email,
            first_name=first_name,
            last_name=last_name,
        )

    def create_user_placeholder(self, email: str) -> UserRecord:
        return self._user_repo.create_user_placeholder(email)

    def get_user_by_subject(self, subject: str) -> Optional[UserRecord]:
        return self._user_repo.get_user_by_subject(subject)

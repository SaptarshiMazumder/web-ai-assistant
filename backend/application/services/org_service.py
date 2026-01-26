from typing import Dict, List, Optional

from domain.entities import OrgMemberRecord, OrgRecord
from domain.repositories import OrgMembershipRepository, OrgRepository


class OrgService:
    def __init__(self, org_repo: OrgRepository, membership_repo: OrgMembershipRepository) -> None:
        self._org_repo = org_repo
        self._membership_repo = membership_repo

    def create_org(self, name: str) -> str:
        return self._org_repo.create_org(name)

    def list_orgs(self) -> List[OrgRecord]:
        return self._org_repo.list_orgs()

    def get_org_by_name(self, name: str) -> Optional[OrgRecord]:
        return self._org_repo.get_org_by_name(name)

    def get_org(self, org_id: str) -> Optional[OrgRecord]:
        return self._org_repo.get_org(org_id)

    def update_org_name(self, org_id: str, name: str) -> None:
        self._org_repo.update_org_name(org_id, name)

    def set_org_status(self, org_id: str, status: str) -> None:
        self._org_repo.set_org_status(org_id, status)

    def add_membership(self, org_id: str, user_id: str, role: str) -> None:
        self._membership_repo.add_membership(org_id, user_id, role)

    def get_org_memberships(self, user_id: str) -> List[Dict[str, str]]:
        return self._membership_repo.get_org_memberships(user_id)

    def list_org_members(self, org_id: str) -> List[OrgMemberRecord]:
        return self._membership_repo.list_org_members(org_id)

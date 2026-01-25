import argparse
import sys

from services.reset_service import delete_gcs_objects, delete_rag_corpora, _parse_bucket_and_prefix, _list_corpora


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all RAG corpora and crawled GCS data.")
    parser.add_argument("--confirm", action="store_true", help="Actually perform deletions.")
    parser.add_argument("--skip-gcs", action="store_true", help="Skip deleting GCS objects.")
    parser.add_argument("--skip-rag", action="store_true", help="Skip deleting RAG corpora.")
    parser.add_argument(
        "--allow-root",
        action="store_true",
        help="Allow deleting the entire bucket when GCS_BUCKET has no prefix.",
    )
    args = parser.parse_args()

    bucket_name, base_prefix = _parse_bucket_and_prefix()
    corpora = _list_corpora()

    print("Planned deletions:")
    if not args.skip_gcs:
        target = f"gs://{bucket_name}/{base_prefix or ''}"
        print(f"  - GCS objects under: {target}")
    if not args.skip_rag:
        print(f"  - RAG corpora: {len(corpora)}")

    if not args.confirm:
        print("\nDry run only. Re-run with --confirm to delete.")
        return 1

    if not args.skip_gcs:
        if not base_prefix and not args.allow_root:
            print("Refusing to delete entire bucket without --allow-root.")
            return 2
        _, _, deleted = delete_gcs_objects(allow_root=args.allow_root)
        print(f"Deleted {deleted} GCS objects.")

    if not args.skip_rag:
        deleted = delete_rag_corpora(corpora)
        print(f"Deleted {deleted} RAG corpora.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

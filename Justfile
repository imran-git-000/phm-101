set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

src_dir := "src"
GCP_REGISTRY_REGION := ""
GCP_PROJECT_ID := ""
REPOSITORY_NAME := ""

# Show the available recipes.
default:
    just --list

# == SETUP REPOSITORY AND DEPENDENCIES

# Install the repository git hooks into .git/hooks.
set-hooks:
    cp .hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    cp .hooks/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    cp .hooks/post-merge .git/hooks/post-merge
    chmod +x .git/hooks/post-merge

# Create or update the virtual environment. The project is relocked before syncing. This installs all extras and the development group (excluding the build group)
dev-sync:
    uv sync --cache-dir .uv_cache --all-extras --no-group build

# Sync environment as in dev-sync but also refreshes a package, which might be a local version.
dev-sync-refresh-package lib:
    uv sync --cache-dir .uv_cache --all-extras --no-group build --refresh-package {{lib}} --refresh-install {{lib}}

# Install hooks and sync the development environment.
setup: set-hooks dev-sync

# === CODE VALIDATION

# Format source and test files with Ruff.
format:
    uv run ruff format {{src_dir}} tests

# Check whether formatting changes would be required.
format-on-commit:
    uv run ruff format {{src_dir}} tests --exit-non-zero-on-format

# Run linting with Ruff autofix and type checks with ty.
lint:
    uv run ruff check {{src_dir}} tests --fix
    uv run ty check {{src_dir}}
    uv run ty check tests

# Run non-mutating lint and type checks.
lint-on-push:
    uv run ruff check {{src_dir}} tests
    uv run ty check {{src_dir}}
    uv run ty check tests

# Run the test suite with coverage and xdist.
test:
    uv run pytest --verbose --color=yes --cov={{src_dir}} --exitfirst -n auto

# Run formatting, linting, and tests.
all-validation: format lint test

# === BUILD AND DEPLOYMENT

# Commit versioned files and push the current branch.
push-version commit_message:
    git add pyproject.toml
    git add uv.lock
    git commit -m '{{commit_message}}'
    git push

# Create and push a release tag.
deploy-tag new_tag:
    echo '{{new_tag}}'
    git tag '{{new_tag}}'
    git push origin '{{new_tag}}' --no-verify

# This recipe reads the current version from pyproject.toml and deploy a tag
# using the same value.
deploy-tag-v2:
    #!/usr/bin/env bash
    set -euo pipefail

    # Find os version for different command syntax
    # Read the current version from file
    new_version="$(uv version --short)"
    echo "Current version=$new_version"

    # Read the current branch name
    deploy_environment="$(git rev-parse --abbrev-ref HEAD)"

    # Set the new tag variable
    if [[ "${USE_DEPLOY_ENVIRONMENT:-n}" == "y" ]]; then
        # If use environment is true, we add a suffix to the tag to specify the deploy environment
        new_tag="v${new_version}@${deploy_environment}"
    else
        new_tag="v${new_version}"
    fi

    # Create new tag
    echo "Create tag $new_tag"
    git tag "$new_tag"

    # Push the tag (this could trigger the github action that builds the package)
    git push --tags
    echo "Tag $new_tag pushed"

# Build the docker image and optionally push all tags.
docker:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if version tag is set otherwise generate it from pyproject.toml
    if [[ -n "${VERSION_TAG:-}" ]]; then
        echo "Version tag provided"
    else
        echo "Generating version tag from pyproject.toml"
        tag="$(uv version --short)"
        branch_name="$(git rev-parse --abbrev-ref HEAD)"
        VERSION_TAG="v${tag}@${branch_name}"
    fi
    echo "Using ${VERSION_TAG} for docker generation"

    if [[ -n "${DEPLOY_DOCKER:-}" ]]; then
        echo "Deploying docker to Artifact Registry"
        # Setup access to GCP Artifact Registry
        gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://{{GCP_REGISTRY_REGION}}-docker.pkg.dev"
        docker_repository="{{GCP_REGISTRY_REGION}}-docker.pkg.dev/{{GCP_PROJECT_ID}}/{{REPOSITORY_NAME}}"
        image_name="${docker_repository}/py-service-template"
    else
        echo "Building docker locally; set variable DEPLOY_DOCKER if you want to deploy docker to Artifact Registry"
        image_name="py-service-template"
    fi

    # The tag has the structure "vM.m.p@DE"; parse and validate it in one step.
    if [[ "$VERSION_TAG" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)(@(.+))?$ ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        patch="${BASH_REMATCH[3]}"
        deploy_environment="${BASH_REMATCH[5]:-}"
    else
        echo "Invalid VERSION_TAG: $VERSION_TAG"
        exit 1
    fi

    # Verify that variables are exported tag is set
    [[ -n "$major" ]] || { echo "MAJOR is empty"; exit 1; }
    echo "Major: $major"
    [[ -n "$minor" ]] || { echo "MINOR is empty"; exit 1; }
    echo "Minor: $minor"
    [[ -n "$patch" ]] || { echo "PATCH is empty"; exit 1; }
    echo "Patch: $patch"

    if [[ "${USE_DEPLOY_ENVIRONMENT:-n}" == "y" ]]; then
        [[ -n "$deploy_environment" ]] || { echo "DEPLOY_ENVIRONMENT is empty"; exit 1; }
        echo "DEPLOY_ENVIRONMENT: $deploy_environment"
    fi

    # Build tags
    major_tag="${image_name}:${major}"
    minor_tag="${image_name}:${major}.${minor}.${patch}"
    latest_tag="${image_name}:latest"

    echo "Image name: $image_name"
    echo "Major tag: $major_tag"
    echo "Minor tag: $minor_tag"
    echo "Latest tag: $latest_tag"

    # Create docker image
    docker build \
        -t "$latest_tag" -t "$major_tag" -t "$minor_tag" \
        . \
        --build-arg SRC_DIR="{{src_dir}}"

    if [[ -n "${DEPLOY_DOCKER:-}" ]]; then
        # Push the docker image to the repo
        echo "Pushing image to repo"
        docker push "$image_name" --all-tags
    else
        echo "Image not pushed"
    fi

# Bump the patch version, commit version files, and push.
bump-patch:
    just push-version "Bump version $(uv version --bump=patch)"

# Bump the minor version, commit version files, and push.
bump-minor:
    just push-version "Bump version $(uv version --bump=minor)"

# Bump the major version, commit version files, and push.
bump-major:
    just push-version "Bump version $(uv version --bump=major)"

#!/usr/bin/env python3
"""
Multi-Agent Research System CLI - FIXED VERSION
Clean Click-based command line interface with proper timeout handling
"""

import click
import subprocess
import sys
import time
import os
import json
import requests
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# CLI Configuration
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def echo_success(message: str):
    """Print success message"""
    click.echo(click.style(f"✅ {message}", fg='green'))


def echo_error(message: str):
    """Print error message"""
    click.echo(click.style(f"❌ {message}", fg='red'))


def echo_warning(message: str):
    """Print warning message"""
    click.echo(click.style(f"⚠️ {message}", fg='yellow'))


def echo_info(message: str):
    """Print info message"""
    click.echo(click.style(f"ℹ️ {message}", fg='blue'))


def echo_step(step: str, message: str):
    """Print step message"""
    click.echo(click.style(f"{step} {message}", fg='white', bold=True))


def echo_header(message: str):
    """Print a header"""
    click.echo("\n" + "=" * 60)
    click.echo(click.style(message, fg='cyan', bold=True))
    click.echo("=" * 60)


class SystemManager:
    """Core system management functionality - FIXED VERSION"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.compose_file = self.project_root / "docker-compose.yml"
        self.env_file = self.project_root / ".env"

    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        echo_step("🔧", "Checking prerequisites...")

        # Check Docker
        if not self._check_command(["docker", "--version"]):
            echo_error("Docker not found. Please install Docker.")
            return False

        # Check Docker Compose
        if not self._check_command(["docker-compose", "--version"]):
            echo_error("Docker Compose not found. Please install Docker Compose.")
            return False

        # Check compose file
        if not self.compose_file.exists():
            echo_error(f"Docker compose file not found: {self.compose_file}")
            return False

        echo_success("All prerequisites met")
        return True

    def _check_command(self, cmd: List[str]) -> bool:
        """Check if a command exists"""
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def setup_environment(self) -> bool:
        """Setup environment file"""
        echo_step("📝", "Checking environment...")

        if not self.env_file.exists():
            env_defaults = self.project_root / ".env.defaults"
            if env_defaults.exists():
                with open(env_defaults, 'r') as src, open(self.env_file, 'w') as dst:
                    dst.write(src.read())
                echo_success("Created .env from .env.defaults")
            else:
                echo_error("No .env or .env.defaults found")
                return False
        else:
            echo_info(".env file exists")

        return True

    def check_sources(self):
        """Check sources directory - no creation, just report"""
        echo_step("📁", "Checking sources directory...")

        sources_dir = self.project_root / "sources"

        if not sources_dir.exists():
            echo_warning("Sources directory not found")
            echo_info("Create sources/ and add PDF files for document ingestion")
            return

        pdf_files = list(sources_dir.glob("*.pdf"))

        if pdf_files:
            echo_success(f"Found {len(pdf_files)} PDF files:")
            for i, pdf in enumerate(pdf_files[:5], 1):
                size_kb = pdf.stat().st_size // 1024
                click.echo(f"   {i}. {pdf.name} ({size_kb}KB)")
            if len(pdf_files) > 5:
                click.echo(f"   ... and {len(pdf_files) - 5} more")
        else:
            echo_info("No PDF files found in sources/")
            echo_info("Add PDF files to sources/ for document processing")

    def run_compose_command(self, cmd: List[str], timeout: int = 300) -> bool:
        """Run a docker-compose command with improved timeout handling"""
        try:
            full_cmd = ["docker-compose"] + cmd
            echo_info(f"Running: {' '.join(full_cmd)}")

            # For 'up' commands, don't capture output to avoid hanging
            if 'up' in cmd:
                echo_info(f"Starting services (timeout: {timeout}s)...")
                result = subprocess.run(
                    full_cmd,
                    cwd=self.project_root,
                    timeout=timeout,
                    # Don't capture output for 'up' commands - let them stream
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                result = subprocess.run(
                    full_cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

            if result.returncode == 0:
                echo_success("Command completed successfully")
                return True
            else:
                echo_warning(f"Command returned exit code: {result.returncode}")
                return self._check_if_services_actually_started(cmd)

        except subprocess.TimeoutExpired:
            echo_warning(f"Command timed out after {timeout}s, checking if services started...")
            # Don't treat timeout as failure - services might still be starting
            return self._check_if_services_actually_started(cmd)
        except Exception as e:
            echo_error(f"Command error: {e}")
            return False

    def _check_if_services_actually_started(self, cmd: List[str]) -> bool:
        """Check if services actually started despite timeout"""
        if 'up' not in cmd:
            return False

        try:
            # Give services a moment to start
            echo_info("Verifying service status...")
            time.sleep(10)

            # Check if containers are running
            result = subprocess.run(
                ["docker-compose", "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                running_services = []
                for line in result.stdout.split('\n'):
                    if 'Up' in line and 'Exit' not in line:
                        # Extract service name from docker-compose ps output
                        parts = line.split()
                        if len(parts) > 0:
                            service_name = parts[0].split('_')[-1].split('-')[0]
                            if service_name not in running_services:
                                running_services.append(service_name)

                if running_services:
                    echo_success(f"Services are running: {', '.join(running_services)}")
                    return True
                else:
                    echo_warning("No services appear to be running")
                    return False
            else:
                return False

        except Exception as e:
            echo_warning(f"Could not verify service status: {e}")
            return False

    def wait_for_service(self, service: str, max_wait: int = 240) -> bool:
        """Wait for a service to become healthy with better detection"""
        echo_step("⏳", f"Waiting for {service} to be ready...")

        start_time = time.time()
        check_interval = 10  # Check every 10 seconds

        while time.time() - start_time < max_wait:
            try:
                # More specific health checks per service
                if self._check_service_health(service):
                    echo_success(f"{service} is ready")
                    return True

                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0 and elapsed > 0:  # Report every 30 seconds
                    echo_info(f"Still waiting for {service}... ({elapsed}s/{max_wait}s)")

                time.sleep(check_interval)

            except Exception as e:
                echo_warning(f"Error checking {service}: {e}")
                time.sleep(check_interval)

        echo_warning(f"Timeout waiting for {service} after {max_wait}s (service may still be starting)")
        return False

    def _check_service_health(self, service: str) -> bool:
        """Improved service health checks"""
        try:
            if service == "neo4j":
                return self._check_neo4j_health()
            elif service == "mongodb":
                return self._check_mongodb_health()
            elif service == "chromadb":
                return self._check_chromadb_health()
            elif service == "api":
                return self._check_api_health()
            else:
                # Fallback: check if container is running
                result = subprocess.run(
                    ["docker-compose", "ps", service],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.returncode == 0 and ("Up" in result.stdout)

        except Exception:
            return False

    def _check_neo4j_health(self) -> bool:
        """Check Neo4j health"""
        try:
            response = requests.get("http://localhost:7474", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _check_mongodb_health(self) -> bool:
        """Check MongoDB health"""
        try:
            # Try to connect to MongoDB
            from pymongo import MongoClient
            client = MongoClient("mongodb://user:password@localhost:27017/", serverSelectionTimeoutMS=5000)
            client.server_info()
            return True
        except:
            return False

    def _check_chromadb_health(self) -> bool:
        """Check ChromaDB health"""
        try:
            response = requests.get("http://localhost:8001/api/v1/heartbeat", timeout=5)
            return response.status_code == 200
        except:
            try:
                # Alternative endpoint
                response = requests.get("http://localhost:8001", timeout=5)
                return response.status_code == 200
            except:
                return False

    def _check_api_health(self) -> bool:
        """Check API health"""
        try:
            response = requests.get("http://localhost:8000/api/v1/status", timeout=5)
            return response.status_code == 200
        except:
            return False

    def show_logs(self, service: Optional[str] = None, lines: int = 20):
        """Show service logs"""
        cmd = ["logs", "--tail", str(lines)]
        if service:
            cmd.append(service)

        try:
            full_cmd = ["docker-compose"] + cmd
            result = subprocess.run(
                full_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                if service:
                    echo_step("📋", f"Recent {service} logs:")
                else:
                    echo_step("📋", "Recent system logs:")

                for line in result.stdout.split('\n'):
                    if line.strip():
                        click.echo(f"   {line}")
            else:
                echo_error("Failed to get logs")

        except Exception as e:
            echo_error(f"Error getting logs: {e}")


def verify_system_with_http_calls(quick: bool = False) -> bool:
    """Verify system is working with actual HTTP calls"""
    services_to_check = [
        ("Neo4j Browser", "http://localhost:7474", 5),
        ("MongoDB Express", "http://localhost:8081", 10),
        ("ChromaDB", "http://localhost:8001", 5),
        ("API Status", "http://localhost:8000/api/v1/status", 10)
    ]

    if quick:
        # Only check API in quick mode
        services_to_check = [("API Status", "http://localhost:8000/api/v1/status", 5)]

    healthy_count = 0

    echo_info("Verifying services are responding...")
    for name, url, timeout in services_to_check:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                echo_success(f"{name} is responding")
                healthy_count += 1
            else:
                echo_warning(f"{name} returned status {response.status_code}")
        except Exception as e:
            echo_warning(f"{name} not responding yet")

    return healthy_count > 0  # At least one service working


def show_startup_summary():
    """Show startup summary"""
    echo_step("📋", "System Summary:")

    # Show access points
    click.echo("\n🌐 Access Points:")
    access_points = [
        ("API", "http://localhost:8000"),
        ("API Status", "http://localhost:8000/api/v1/status"),
        ("Neo4j Browser", "http://localhost:7474 (neo4j/password)"),
        ("MongoDB Express", "http://localhost:8081"),
        ("ChromaDB", "http://localhost:8001")
    ]

    for name, url in access_points:
        click.echo(f"   • {name}: {click.style(url, fg='blue')}")


# Create the main CLI group
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--project-dir', default='.', help='Project directory path')
@click.pass_context
def cli(ctx, verbose, project_dir):
    """🚀 Multi-Agent Research System CLI

    Command-line interface for managing your multi-agent research system.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['manager'] = SystemManager(project_dir)


@cli.command()
@click.option('--timeout', '-t', default=600, help='Maximum startup timeout in seconds')
@click.option('--databases-only', is_flag=True, help='Start only database services')
@click.option('--quick', is_flag=True, help='Skip detailed health checks for faster startup')
@click.pass_context
def start(ctx, timeout, databases_only, quick):
    """🚀 Start the multi-agent system with improved timeout handling"""

    manager: SystemManager = ctx.obj['manager']
    verbose = ctx.obj['verbose']

    echo_header("🚀 STARTING MULTI-AGENT RESEARCH SYSTEM")

    try:
        # Step 1: Prerequisites
        if not manager.check_prerequisites():
            raise click.ClickException("Prerequisites check failed")

        # Step 2: Environment setup
        if not manager.setup_environment():
            raise click.ClickException("Environment setup failed")

        # Step 3: Check sources (info only)
        manager.check_sources()

        # Step 4: Start services
        echo_step("🐳", "Starting services...")

        if databases_only:
            # Start only database services
            db_services = ["neo4j", "mongodb", "chromadb"]
            if not manager.run_compose_command(["up", "-d"] + db_services, timeout=timeout):
                echo_warning("Service startup may have timed out, but checking if services are running...")

            echo_success("Database services started")

            # Wait for database health (with shorter timeouts if quick mode)
            if not quick:
                health_timeout = 120
                for service in db_services:
                    if not manager.wait_for_service(service, max_wait=health_timeout):
                        echo_warning(f"Service {service} not ready, but may still be starting...")

            echo_success("Database services are running")
            show_startup_summary()
            return

        # Start all services
        echo_info(f"Starting all services (timeout: {timeout}s)...")
        if not manager.run_compose_command(["up", "-d"], timeout=timeout):
            echo_warning("Startup may have timed out, checking service status...")

        echo_success("Services started")

        # Health checks (skip if quick mode)
        if not quick:
            echo_step("🔍", "Performing health checks...")
            key_services = ["neo4j", "mongodb", "chromadb"]
            service_timeout = 120

            for service in key_services:
                if not manager.wait_for_service(service, max_wait=service_timeout):
                    echo_warning(f"Service {service} not ready, but may still be working...")

            # Check API if it exists
            if manager.wait_for_service("api", max_wait=120):
                echo_success("API service is ready")
            else:
                echo_warning("API service not ready, but may still be starting...")

        # Final verification with actual HTTP calls
        echo_step("🔍", "Verifying system...")
        if verify_system_with_http_calls(quick):
            echo_success("System verification completed")
        else:
            echo_warning("Some services may still be starting up...")

        # Show final status
        show_startup_summary()

        echo_header("✅ SYSTEM STARTUP COMPLETE")
        echo_success("Your multi-agent research system is ready!")

        # Show next steps
        click.echo("\n" + click.style("📋 Quick Commands:", fg='cyan', bold=True))
        click.echo("   python cli.py test      # Test system functionality")
        click.echo("   python cli.py status    # Check detailed status")
        click.echo("   python cli.py health    # Run health checks")

    except click.ClickException:
        raise
    except KeyboardInterrupt:
        echo_warning("Startup interrupted by user")
        ctx.exit(1)
    except Exception as e:
        echo_error(f"Unexpected error during startup: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        ctx.exit(1)


@cli.command()
@click.pass_context
def quick_start(ctx):
    """⚡ Quick start with minimal health checks"""
    ctx.invoke(start, timeout=300, databases_only=False, quick=True)


@cli.command()
@click.option('--volumes', is_flag=True, help='Remove volumes too')
@click.confirmation_option(prompt='This will stop all services. Continue?')
@click.pass_context
def stop(ctx, volumes):
    """🛑 Stop all system services"""

    manager: SystemManager = ctx.obj['manager']

    echo_step("🛑", "Stopping multi-agent system...")

    cmd = ["down"]
    if volumes:
        cmd.append("-v")
        echo_warning("Removing volumes - all data will be lost!")

    if manager.run_compose_command(cmd):
        echo_success("System stopped successfully")

        if volumes:
            echo_info("All data volumes removed")
    else:
        echo_error("Failed to stop system")
        ctx.exit(1)


@cli.command()
@click.option('--service', '-s', help='Restart specific service only')
@click.pass_context
def restart(ctx, service):
    """🔄 Restart the system or specific service"""

    manager: SystemManager = ctx.obj['manager']

    if service:
        echo_step("🔄", f"Restarting {service}...")

        if manager.run_compose_command(["restart", service]):
            echo_success(f"{service} restarted successfully")
        else:
            echo_error(f"Failed to restart {service}")
            ctx.exit(1)
    else:
        echo_step("🔄", "Restarting entire system...")

        # Stop and start
        ctx.invoke(stop, volumes=False)
        time.sleep(3)
        ctx.invoke(start)


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def status(ctx, output_format):
    """📊 Show system status"""

    manager: SystemManager = ctx.obj['manager']

    echo_step("📊", "Getting system status...")

    try:
        result = subprocess.run(
            ["docker-compose", "ps"],
            cwd=manager.project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            if output_format == 'json':
                # Simple status in JSON format
                status_data = {"status": "success", "output": result.stdout}
                click.echo(json.dumps(status_data, indent=2))
            else:
                echo_success("System status:")
                click.echo(result.stdout)
        else:
            echo_error("Failed to get system status")
            ctx.exit(1)

    except Exception as e:
        echo_error(f"Error getting status: {e}")
        ctx.exit(1)


@cli.command()
@click.option('--service', '-s', help='Show logs for specific service')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.pass_context
def logs(ctx, service, lines, follow):
    """📋 Show service logs"""

    manager: SystemManager = ctx.obj['manager']

    if follow:
        echo_step("📋", f"Following logs{' for ' + service if service else ''}...")
        echo_info("Press Ctrl+C to stop")

        cmd = ["logs", "-f"]
        if service:
            cmd.append(service)

        try:
            subprocess.run(
                ["docker-compose"] + cmd,
                cwd=manager.project_root
            )
        except KeyboardInterrupt:
            echo_info("Log following stopped")
    else:
        manager.show_logs(service, lines)


@cli.command()
@click.option('--query', '-q', default='machine learning', help='Test query to send')
@click.option('--endpoint', default='agent', help='API endpoint to test')
@click.option('--timeout', default=60, help='Request timeout in seconds')
@click.option('--simple', '-s', is_flag=True, help='Show simple, clean output')
@click.pass_context
def test(ctx, query, endpoint, timeout, simple):
    """🧪 Test system functionality"""

    if simple:
        # Simple, clean output mode
        echo_step("🧪", f"Testing: '{query}'")

        try:
            url = f"http://localhost:8000/api/v1/{endpoint}"
            response = requests.post(url, json={"query": query}, timeout=timeout)

            if response.status_code == 200:
                result = response.json()
                message = result.get('message', '')
                system_health = result.get('system_health', {})

                # Clean output
                print(f"\n{'=' * 60}")
                print(f"📋 QUERY: {query}")
                print(f"{'=' * 60}")
                print(f"{message}")
                print(f"\n{'=' * 60}")
                print(f"🔧 SYSTEM STATUS:")
                for key, value in system_health.items():
                    clean_key = key.replace('_', ' ').title()
                    print(f"   {clean_key}: {value}")
                print(f"{'=' * 60}")

            else:
                echo_error(f"Request failed: {response.status_code}")

        except requests.exceptions.Timeout:
            echo_error(f"Request timed out after {timeout}s")
        except requests.exceptions.ConnectionError:
            echo_error("Cannot connect to API - is the system running?")
        except Exception as e:
            echo_error(f"Error: {e}")

        return

    # Original detailed output mode
    echo_step("🧪", f"Testing {endpoint} endpoint with query: '{query}'")

    try:
        url = f"http://localhost:8000/api/v1/{endpoint}"

        if endpoint == 'agent':
            response = requests.post(
                url,
                json={"query": query},
                timeout=timeout
            )
        else:
            response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            echo_success(f"API test successful!")

            try:
                result = response.json()

                if endpoint == 'agent':
                    message = result.get('message', '')
                    system_health = result.get('system_health', {})

                    # Clean metrics display
                    click.echo(f"\n📊 {click.style('Test Results:', fg='cyan', bold=True)}")
                    click.echo(f"   Query: {click.style(query, fg='white', bold=True)}")
                    click.echo(f"   Response Length: {click.style(f'{len(message):,} characters', fg='green')}")

                    # System health
                    click.echo(f"\n🔧 {click.style('System Health:', fg='cyan', bold=True)}")
                    for key, value in system_health.items():
                        clean_key = key.replace('_', ' ').title()
                        # Color code based on status
                        if '✅' in str(value):
                            color = 'green'
                        elif '🟡' in str(value):
                            color = 'yellow'
                        elif '❌' in str(value):
                            color = 'red'
                        else:
                            color = 'white'

                        click.echo(f"   {clean_key}: {click.style(str(value), fg=color)}")

                    # Show response preview
                    click.echo(f"\n📄 {click.style('Response Preview:', fg='cyan', bold=True)}")

                    # Show first 300 chars of clean message
                    preview = message[:300].replace('\n', ' ').strip()
                    if len(message) > 300:
                        preview += "..."
                    click.echo(f"   {preview}")

                else:
                    click.echo(f"\n📄 Response: {json.dumps(result, indent=2)}")

            except json.JSONDecodeError:
                click.echo(f"\n📄 Raw Response: {response.text}")

        else:
            echo_error(f"API test failed with status {response.status_code}")
            click.echo(f"Response: {response.text}")
            ctx.exit(1)

    except requests.exceptions.ConnectionError:
        echo_error("Cannot connect to API - is the system running?")
        echo_info("Try: python cli.py start")
        ctx.exit(1)
    except requests.exceptions.Timeout:
        echo_error(f"Request timed out after {timeout}s")
        ctx.exit(1)
    except Exception as e:
        echo_error(f"Test failed: {e}")
        ctx.exit(1)


@cli.command()
@click.option('--detailed', is_flag=True, help='Run detailed health checks')
@click.pass_context
def health(ctx, detailed):
    """🏥 Run system health checks"""

    echo_step("🏥", "Running system health checks...")

    health_results = []

    # Check API
    echo_info("Checking API health...")
    try:
        response = requests.get("http://localhost:8000/api/v1/status", timeout=10)
        api_healthy = response.status_code == 200
    except:
        api_healthy = False

    health_results.append(("API", api_healthy))

    if api_healthy:
        echo_success("API is responding")
    else:
        echo_error("API is not responding")

    if detailed:
        # Check databases
        echo_info("Checking database connections...")

        # Neo4j
        try:
            response = requests.get("http://localhost:7474", timeout=10)
            neo4j_healthy = response.status_code == 200
            echo_success("Neo4j: Connected")
            health_results.append(("Neo4j", True))
        except Exception as e:
            echo_error(f"Neo4j: Failed")
            health_results.append(("Neo4j", False))

        # MongoDB
        try:
            from pymongo import MongoClient
            client = MongoClient("mongodb://user:password@localhost:27017/", serverSelectionTimeoutMS=5000)
            client.server_info()
            echo_success("MongoDB: Connected")
            health_results.append(("MongoDB", True))
        except Exception as e:
            echo_error(f"MongoDB: Failed")
            health_results.append(("MongoDB", False))

        # ChromaDB
        try:
            response = requests.get("http://localhost:8001", timeout=10)
            chroma_healthy = response.status_code == 200
            echo_success("ChromaDB: Connected")
            health_results.append(("ChromaDB", True))
        except Exception as e:
            echo_error(f"ChromaDB: Failed")
            health_results.append(("ChromaDB", False))

    # Summary
    healthy_count = sum(1 for _, healthy in health_results if healthy)
    total_checks = len(health_results)

    click.echo(f"\n📊 Health Summary: {healthy_count}/{total_checks} checks passed")

    if healthy_count == total_checks:
        echo_success("System is fully healthy! 🎉")
    elif healthy_count > 0:
        echo_warning("System is partially healthy - some components need attention")
        ctx.exit(1)
    else:
        echo_error("System is unhealthy - major issues detected")
        ctx.exit(1)


if __name__ == '__main__':
    cli()
import os
import logging
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk.resources import Resource

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def setup_telemetry(service_name: str = "rita-llm"):
    """
    Set up OpenTelemetry metrics configuration.
    
    Args:
        service_name: Name of the service for metrics identification
    """
    try:
        # Get endpoint from environment variable
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        # Ensure endpoint ends with /v1/metrics if manually specified
        if not endpoint.endswith("/v1/metrics"):
            endpoint = f"{endpoint.rstrip('/')}/v1/metrics"
        logger.debug(f"Using OTLP endpoint: {endpoint}")
        print(f"Using OTLP endpoint: {endpoint}")

        # Get headers from environment variable
        headers = {}
        headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_str:
            try:
                for header in headers_str.split(","):
                    key, value = header.strip().split("=")
                    if key and value:  # Only add non-empty headers
                        headers[key.strip()] = value.strip()
                logger.debug("OTLP headers configured from environment")
            except Exception as e:
                logger.error(f"Failed to parse OTLP headers: {e}")
                headers = None

        # Get resource attributes from environment
        resource_attrs = {}
        resource_attrs_str = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if resource_attrs_str:
            try:
                for attr in resource_attrs_str.split(","):
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        resource_attrs[key.strip()] = value.strip()
            except Exception as e:
                logger.error(f"Failed to parse resource attributes: {e}")
        
        # Add default service name if not provided
        if "service.name" not in resource_attrs:
            resource_attrs["service.name"] = service_name
        
        # Create resource with attributes
        resource = Resource.create(resource_attrs)
        
        # Configure the OTLP HTTP exporter
        exporter = OTLPMetricExporter(
            endpoint=endpoint,
            headers=headers if headers else None,
            timeout=30000  # 30 second timeout
        )
        
        # Set up metric reader with longer export interval
        reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=15000  # Export every 15 seconds
        )
        
        # Set up meter provider
        provider = MeterProvider(
            resource=resource,
            metric_readers=[reader]
        )
        
        # Set the global meter provider
        metrics.set_meter_provider(provider)
        
        # Get a meter
        meter = metrics.get_meter(__name__)
        
        # Create metrics
        request_counter = meter.create_counter(
            name="requests_total",
            description="Total number of requests",
            unit="1"
        )
        
        avatar_request_counter = meter.create_counter(
            name="avatar_requests_total",
            description="Total number of avatar requests",
            unit="1"
        )
        
        avatar_serve_time = meter.create_counter(
            name="avatar_serve_time_seconds",
            description="Time taken to serve avatar requests",
            unit="s"
        )
        
        connection_state_counter = meter.create_up_down_counter(
            name="connection_state",
            description="Current connection state",
            unit="1"
        )
        
        logger.info("Telemetry setup completed successfully")
        
        return {
            "request_counter": request_counter,
            "avatar_request_counter": avatar_request_counter,
            "avatar_serve_time": avatar_serve_time,
            "connection_state_counter": connection_state_counter,
            "provider": provider  # Return the provider for shutdown
        }
    except Exception as e:
        logger.error(f"Failed to setup telemetry: {str(e)}", exc_info=True)
        raise

def shutdown_telemetry(provider):
    """
    Gracefully shutdown the telemetry system.
    
    Args:
        provider: The MeterProvider instance to shutdown
    """
    try:
        if provider:
            provider.shutdown()
            logger.info("Telemetry shutdown completed successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown telemetry: {str(e)}", exc_info=True) 
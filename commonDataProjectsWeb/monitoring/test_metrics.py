#!/usr/bin/env python3
"""
Test script to verify Prometheus metrics are being exposed correctly.
Run this script to check if Django metrics endpoint is accessible.
"""
import requests
import sys

def test_django_metrics():
    """Test Django metrics endpoint."""
    url = "http://localhost:8000/metrics/"
    
    try:
        print(f"Testing Django metrics endpoint: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Metrics endpoint is accessible (Status: {response.status_code})")
            print(f"   Response length: {len(response.text)} bytes")
            
            # Check for Django Prometheus metrics
            content = response.text
            django_metrics = [
                'django_http_requests_total',
                'django_http_request_duration_seconds',
                'django_db_',
                'python_info'
            ]
            
            found_metrics = []
            for metric in django_metrics:
                if metric in content:
                    found_metrics.append(metric)
                    print(f"   ✅ Found metric: {metric}")
            
            if found_metrics:
                print(f"\n✅ Found {len(found_metrics)} Django Prometheus metrics")
                return True
            else:
                print("\n⚠️  Metrics endpoint accessible but no Django metrics found")
                print("   This might mean no HTTP requests have been made yet.")
                print("   Try accessing your Django app (http://localhost:8000) first.")
                return False
        else:
            print(f"❌ Metrics endpoint returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {url}")
        print("   Make sure Django is running: docker-compose up -d web")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prometheus_scrape():
    """Test if Prometheus can scrape Django metrics."""
    url = "http://localhost:9090/api/v1/targets"
    
    try:
        print(f"\nTesting Prometheus targets: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            targets = data.get('data', {}).get('activeTargets', [])
            
            django_targets = [t for t in targets if 'django' in t.get('labels', {}).get('job', '').lower()]
            
            if django_targets:
                target = django_targets[0]
                health = target.get('health', 'unknown')
                last_scrape = target.get('lastScrape', 'never')
                last_error = target.get('lastError', '')
                
                print(f"✅ Found Django target in Prometheus")
                print(f"   Health: {health}")
                print(f"   Last scrape: {last_scrape}")
                
                if health == 'up':
                    print(f"   ✅ Target is UP - metrics are being scraped")
                    return True
                elif health == 'down':
                    print(f"   ❌ Target is DOWN")
                    if last_error:
                        print(f"   Error: {last_error}")
                    return False
                else:
                    print(f"   ⚠️  Target health: {health}")
                    return False
            else:
                print("❌ No Django target found in Prometheus")
                print("   Check Prometheus configuration")
                return False
        else:
            print(f"❌ Prometheus API returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Prometheus at {url}")
        print("   Make sure Prometheus is running: docker-compose up -d prometheus")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prometheus_query():
    """Test if Prometheus has collected Django metrics."""
    url = "http://localhost:9090/api/v1/query"
    params = {'query': 'django_http_requests_total'}
    
    try:
        print(f"\nTesting Prometheus query for django_http_requests_total")
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('data', {}).get('result', [])
            
            if result:
                print(f"✅ Prometheus has collected {len(result)} metric series")
                print(f"   Sample metrics:")
                for i, metric in enumerate(result[:3]):
                    metric_name = metric.get('metric', {})
                    print(f"   - {metric_name.get('__name__', 'unknown')} {metric_name}")
                return True
            else:
                print("⚠️  Prometheus is running but has no Django metrics yet")
                print("   This is normal if:")
                print("   1. Django app hasn't received any HTTP requests")
                print("   2. Prometheus just started and hasn't scraped yet")
                print("   Try: Access http://localhost:8000 and wait 15-30 seconds")
                return False
        else:
            print(f"❌ Prometheus query API returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Prometheus")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Prometheus & Grafana Metrics Test")
    print("=" * 60)
    
    # Test Django metrics endpoint
    django_ok = test_django_metrics()
    
    # Test Prometheus scraping
    prometheus_ok = test_prometheus_scrape()
    
    # Test Prometheus query
    query_ok = test_prometheus_query()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Django metrics endpoint: {'✅ OK' if django_ok else '❌ FAIL'}")
    print(f"  Prometheus scraping: {'✅ OK' if prometheus_ok else '❌ FAIL'}")
    print(f"  Prometheus has data: {'✅ OK' if query_ok else '⚠️  No data yet'}")
    print("=" * 60)
    
    if not django_ok:
        print("\n💡 To fix:")
        print("   1. Make sure Django is running: docker-compose up -d web")
        print("   2. Check Django logs: docker-compose logs web")
    elif not prometheus_ok:
        print("\n💡 To fix:")
        print("   1. Restart Prometheus: docker-compose restart prometheus")
        print("   2. Check Prometheus logs: docker-compose logs prometheus")
        print("   3. Verify Prometheus config: http://localhost:9090/config")
    elif not query_ok:
        print("\n💡 To generate metrics:")
        print("   1. Access your Django app: http://localhost:8000")
        print("   2. Make some HTTP requests (navigate pages)")
        print("   3. Wait 15-30 seconds for Prometheus to scrape")
        print("   4. Check Grafana again")
    
    sys.exit(0 if (django_ok and prometheus_ok) else 1)




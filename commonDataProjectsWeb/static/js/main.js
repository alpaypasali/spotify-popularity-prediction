// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
    console.log('Django Microservice - JavaScript loaded');
    initApp();
});

function initApp() {
    console.log('App initialized');
}

// Utility function for API calls
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call error:', error);
        throw error;
    }
}

// Export for use in other scripts
window.apiCall = apiCall;

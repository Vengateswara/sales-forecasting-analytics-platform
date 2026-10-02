// frontend/js/dashboard.js
// Dashboard functionality for Sales Forecasting Platform

// ===== EDA Status =====
function updateEDAStatus(message, type = 'info') {
    // No longer needed - removed status badge
    console.log(`EDA: ${message}`);
}

// ===== Clear EDA =====
function clearEDA() {
    const container = document.getElementById('edaCharts');
    if (!container) return;
    
    container.innerHTML = `
        <div class="alert alert-info">
            <i class="fas fa-info-circle me-2"></i>
            Click <strong>"Generate EDA"</strong> to explore your data. Make sure you have completed <strong>Feature Engineering</strong> first.
        </div>
    `;
    showToast('EDA cleared', 'info');
}

// ===== Load EDA Charts =====
async function loadEDACharts() {
    const container = document.getElementById('edaCharts');
    if (!container) return;
    
    // Check if we have engineered data
    if (!appState.engineered_data && !appState.featureEngineering) {
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Please complete <strong>Feature Engineering</strong> first before generating EDA visualizations.
                <br><br>
                <a href="#" onclick="showSection('features')" class="btn btn-primary btn-sm">
                    <i class="fas fa-cubes me-1"></i>Go to Feature Engineering
                </a>
            </div>
        `;
        return;
    }
    
    showLoading('Loading EDA visualizations...', 10);
    
    try {
        updateProgress(30);
        const response = await fetch('/api/eda');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        updateProgress(60);
        const result = await response.json();
        
        updateProgress(90);
        if (result.status === 'success') {
            appState.edaData = result.data;
            displayEDA(result.data);
            showToast('EDA visualizations loaded!', 'success');
        } else {
            throw new Error(result.message || 'Failed to load EDA');
        }
    } catch (error) {
        console.error('EDA error:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong>Error loading EDA:</strong> ${error.message}
                <br><br>
                <button class="btn btn-outline-danger btn-sm" onclick="loadEDACharts()">
                    <i class="fas fa-redo me-1"></i>Retry
                </button>
            </div>
        `;
        showToast(`EDA error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// ===== Display EDA Results =====
function displayEDA(data) {
    const container = document.getElementById('edaCharts');
    if (!container) return;
    
    // Check if there's an error
    if (data.error) {
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${data.error}
                <br><br>
                <button class="btn btn-outline-warning btn-sm" onclick="loadEDACharts()">
                    <i class="fas fa-redo me-1"></i>Retry
                </button>
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-white">
                        <h6 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Summary Statistics</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${Object.entries(data.summary || {}).map(([key, value]) => `
                                        <tr>
                                            <td>${key.replace('_', ' ').toUpperCase()}</td>
                                            <td>${typeof value === 'number' ? value.toFixed(2) : value}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add trend analysis if available - FULL WIDTH
    if (data.trends && data.trends.chart) {
        html += `
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-white">
                            <h6 class="mb-0"><i class="fas fa-chart-line me-2"></i>Trend Analysis</h6>
                        </div>
                        <div class="card-body">
                            <div id="trendChart"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add distributions if available - FULL WIDTH
    if (data.distributions && data.distributions.length > 0) {
        html += `
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-white">
                            <h6 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Distributions</h6>
                        </div>
                        <div class="card-body">
                            <div id="distChart"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add box plots if available - FULL WIDTH
    if (data.boxplots && data.boxplots.length > 0) {
        html += `
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-white">
                            <h6 class="mb-0"><i class="fas fa-box-plot me-2"></i>Box Plots</h6>
                        </div>
                        <div class="card-body">
                            <div id="boxPlotChart"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add scatter plots if available - FULL WIDTH
    if (data.scatter_plots && data.scatter_plots.length > 0) {
        html += `
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-white">
                            <h6 class="mb-0"><i class="fas fa-scatter-chart me-2"></i>Scatter Plots</h6>
                        </div>
                        <div class="card-body">
                            <div id="scatterChart"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
    
    // Render charts after DOM update
    setTimeout(() => {
        renderEDACharts(data);
    }, 200);
}

// ===== Render EDA Charts =====
function renderEDACharts(data) {
    // Render trend chart - FULL WIDTH
    if (data.trends && data.trends.chart) {
        try {
            const chartData = JSON.parse(data.trends.chart);
            Plotly.newPlot('trendChart', chartData.data, chartData.layout, { 
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering trend chart:', e);
        }
    }
    
    // Render distribution chart - FULL WIDTH
    if (data.distributions && data.distributions.length > 0) {
        try {
            // Combine all distributions into one chart
            const traces = data.distributions.map((dist, index) => {
                const chartData = JSON.parse(dist.chart);
                return chartData.data[0];
            });
            
            const layout = {
                title: 'Feature Distributions',
                xaxis: { title: 'Value' },
                yaxis: { title: 'Frequency' },
                template: 'plotly_white',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                height: 500,
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: 1.02,
                    x: 0.5,
                    xanchor: 'center'
                },
                barmode: 'overlay',
                opacity: 0.7
            };
            
            Plotly.newPlot('distChart', traces, layout, { 
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering distribution chart:', e);
        }
    }
    
    // Render box plot - FULL WIDTH
    if (data.boxplots && data.boxplots.length > 0) {
        try {
            // Combine box plots
            const traces = data.boxplots.map((boxplot) => {
                const chartData = JSON.parse(boxplot.chart);
                return chartData.data[0];
            });
            
            const layout = {
                title: 'Box Plots',
                yaxis: { title: 'Value' },
                template: 'plotly_white',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                height: 500,
                showlegend: false
            };
            
            Plotly.newPlot('boxPlotChart', traces, layout, { 
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering box plot:', e);
        }
    }
    
    // Render scatter plot - FULL WIDTH
    if (data.scatter_plots && data.scatter_plots.length > 0) {
        try {
            const scatter = data.scatter_plots[0];
            const chartData = JSON.parse(scatter.chart);
            Plotly.newPlot('scatterChart', chartData.data, chartData.layout, { 
                responsive: true,
                displayModeBar: true
            });
        } catch (e) {
            console.error('Error rendering scatter plot:', e);
        }
    }
}
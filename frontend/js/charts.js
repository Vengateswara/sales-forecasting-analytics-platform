// frontend/js/charts.js
// Chart rendering utilities for Sales Forecasting Platform

// ===== Create Forecast Chart =====
function createForecastChart(forecast) {
    const container = document.getElementById('forecastChart');
    if (!container) return;
    
    try {
        const dates = forecast.forecast_dates.map(d => new Date(d));
        const values = forecast.forecast_values;
        const lower = forecast.lower_bound || null;
        const upper = forecast.upper_bound || null;
        
        const traces = [];
        
        // Main forecast line
        traces.push({
            x: dates,
            y: values,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Forecast',
            line: { color: '#4a90d9', width: 3 },
            marker: { size: 6, color: '#4a90d9' }
        });
        
        // Confidence interval
        if (lower && upper) {
            traces.push({
                x: dates.concat(dates.slice().reverse()),
                y: lower.concat(upper.slice().reverse()),
                fill: 'toself',
                fillcolor: 'rgba(74, 144, 217, 0.2)',
                line: { color: 'rgba(0,0,0,0)' },
                name: 'Confidence Interval (95%)'
            });
        }
        
        // Add historical data if available
        if (forecast.historical_data) {
            traces.push({
                x: forecast.historical_dates.map(d => new Date(d)),
                y: forecast.historical_data,
                type: 'scatter',
                mode: 'lines',
                name: 'Historical',
                line: { color: '#6c757d', width: 2, dash: 'dash' }
            });
        }
        
        const layout = {
            title: {
                text: 'Sales Forecast',
                font: { size: 18, family: 'Segoe UI, sans-serif' }
            },
            xaxis: { 
                title: 'Date',
                type: 'date',
                tickformat: '%Y-%m-%d'
            },
            yaxis: { 
                title: 'Sales',
                tickprefix: '$'
            },
            hovermode: 'x unified',
            showlegend: true,
            legend: {
                orientation: 'h',
                y: 1.02,
                x: 0.5,
                xanchor: 'center'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 60, r: 30, t: 50, b: 50 }
        };
        
        Plotly.newPlot('forecastChart', traces, layout, { 
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
        
        // Add resize handler
        window.addEventListener('resize', function() {
            Plotly.Plots.resize('forecastChart');
        });
        
    } catch (error) {
        console.error('Error creating forecast chart:', error);
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Error creating forecast chart. Please check your data.
            </div>
        `;
    }
}
function displayTrainingResults(result) {
    const container = document.getElementById('trainingResults');
    if (!container) return;
    
    // Store results in appState for later use
    appState.modelResults = result.results;
    appState.best_model = {
        name: result.best_model,
        testing_score: result.metrics.test_r2,
        cv_score: result.metrics.cv_score || 0,
        metrics: result.metrics,
        training_score: result.metrics.train_r2 || 0
    };
    
    container.innerHTML = `
        <div class="alert alert-success mt-3">
            <i class="fas fa-check-circle me-2"></i>
            ${result.message}
        </div>
        <div class="row mt-3">
            <div class="col-md-4">
                <div class="kpi-card border-success">
                    <div class="kpi-value" style="font-size: 1.2rem;">${result.best_model}</div>
                    <div class="kpi-label">Best Model 🏆</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="kpi-card border-primary">
                    <div class="kpi-value">${(result.metrics.test_r2 * 100).toFixed(2)}%</div>
                    <div class="kpi-label">R² Score</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="kpi-card border-warning">
                    <div class="kpi-value">${result.metrics.test_mae.toFixed(2)}</div>
                    <div class="kpi-label">MAE</div>
                </div>
            </div>
        </div>
        <div class="text-center mt-2">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                Click on <strong>Evaluate</strong> or <strong>Compare</strong> in the sidebar for detailed metrics
            </small>
        </div>
    `;
}


// ===== Create Feature Importance Chart =====
function createFeatureImportanceChart(features, importance) {
    const container = document.getElementById('featureImportanceChart');
    if (!container) return;
    
    try {
        // Sort by importance
        const sorted = features.map((f, i) => ({ feature: f, importance: importance[i] }))
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 15); // Top 15 features
        
        const trace = {
            x: sorted.map(d => d.importance),
            y: sorted.map(d => d.feature),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: sorted.map(d => d.importance / sorted[0].importance * 100),
                colorscale: 'Blues',
                showscale: true,
                colorbar: {
                    title: 'Importance %',
                    tickformat: '.0f'
                }
            },
            text: sorted.map(d => (d.importance / sorted[0].importance * 100).toFixed(1) + '%'),
            textposition: 'outside'
        };
        
        const layout = {
            title: 'Top 15 Feature Importance',
            xaxis: { 
                title: 'Importance Score'
            },
            yaxis: { 
                title: 'Feature',
                automargin: true
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 150, r: 50, t: 50, b: 50 },
            height: 500
        };
        
        Plotly.newPlot('featureImportanceChart', [trace], layout, { 
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
        
    } catch (error) {
        console.error('Error creating feature importance chart:', error);
    }
}

// ===== Create Residual Plot =====
function createResidualPlot(predicted, actual) {
    const container = document.getElementById('residualPlot');
    if (!container) return;
    
    try {
        const residuals = actual.map((a, i) => a - predicted[i]);
        
        const trace1 = {
            x: predicted,
            y: residuals,
            mode: 'markers',
            name: 'Residuals',
            marker: {
                size: 8,
                color: '#4a90d9',
                opacity: 0.6
            }
        };
        
        // Zero line
        const trace2 = {
            x: [Math.min(...predicted), Math.max(...predicted)],
            y: [0, 0],
            mode: 'lines',
            name: 'Zero Line',
            line: {
                color: '#dc3545',
                dash: 'dash',
                width: 2
            }
        };
        
        const layout = {
            title: 'Residual Plot',
            xaxis: { title: 'Predicted Values' },
            yaxis: { title: 'Residuals' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 60, r: 30, t: 50, b: 50 },
            showlegend: true
        };
        
        Plotly.newPlot('residualPlot', [trace1, trace2], layout, { 
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
        
    } catch (error) {
        console.error('Error creating residual plot:', error);
    }
}

// ===== Create Model Comparison Chart (Sorted) =====
function createComparisonChart(results) {
    const container = document.getElementById('comparisonChart');
    if (!container) return;
    
    try {
        // Sort models by testing score (best to worst)
        const sortedModels = Object.entries(results)
            .sort((a, b) => b[1].testing_score - a[1].testing_score);
        
        const models = sortedModels.map(m => m[0]);
        const r2Scores = sortedModels.map(m => m[1].testing_score);
        const maeScores = sortedModels.map(m => m[1].metrics.test_mae);
        const trainTimes = sortedModels.map(m => m[1].train_time);
        
        // R² Score Chart (Best to Worst)
        const trace1 = {
            x: models,
            y: r2Scores.map(v => v * 100),
            type: 'bar',
            name: 'R² Score (%)',
            marker: { 
                color: r2Scores.map((v, i) => i === 0 ? '#28a745' : i === 1 ? '#4a90d9' : i === 2 ? '#ffc107' : '#6c757d')
            },
            text: r2Scores.map(v => (v * 100).toFixed(1) + '%'),
            textposition: 'outside',
            textfont: { size: 10 }
        };
        
        // MAE Chart
        const trace2 = {
            x: models,
            y: maeScores,
            type: 'bar',
            name: 'MAE',
            marker: { 
                color: maeScores.map(v => v === Math.min(...maeScores) ? '#28a745' : '#ffc107')
            },
            text: maeScores.map(v => v.toFixed(2)),
            textposition: 'outside',
            textfont: { size: 10 }
        };
        
        const layout = {
            title: {
                text: 'Model Performance Comparison (Best to Worst)',
                font: { size: 16 }
            },
            xaxis: { 
                title: 'Model',
                tickangle: -45,
                automargin: true
            },
            yaxis: { 
                title: 'Score'
            },
            barmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 60, r: 30, t: 60, b: 150 },
            showlegend: true,
            legend: {
                orientation: 'h',
                y: 1.02,
                x: 0.5,
                xanchor: 'center'
            },
            height: 500,
            hovermode: 'x',
            annotations: [{
                text: '🏆 Best Model',
                x: 0,
                y: 1.05,
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
                font: { size: 12, color: '#28a745' }
            }]
        };
        
        Plotly.newPlot('comparisonChart', [trace1, trace2], layout, { 
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
        
        // Add resize handler
        window.addEventListener('resize', function() {
            Plotly.Plots.resize('comparisonChart');
        });
        
    } catch (error) {
        console.error('Error creating comparison chart:', error);
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Error creating comparison chart: ${error.message}
            </div>
        `;
    }
}

// ===== Create QQ Plot =====
function createQQPlot(data) {
    const container = document.getElementById('qqPlot');
    if (!container) return;
    
    try {
        // Sort data
        const sorted = [...data].sort((a, b) => a - b);
        const n = sorted.length;
        
        // Theoretical quantiles
        const theoretical = Array.from({length: n}, (_, i) => {
            const p = (i + 0.5) / n;
            return Math.sqrt(2) * Math.erf(2 * p - 1);
        });
        
        const trace = {
            x: theoretical,
            y: sorted,
            mode: 'markers',
            name: 'Data',
            marker: {
                size: 8,
                color: '#4a90d9',
                opacity: 0.6
            }
        };
        
        // Reference line
        const minVal = Math.min(...theoretical);
        const maxVal = Math.max(...theoretical);
        const slope = (sorted[sorted.length - 1] - sorted[0]) / (theoretical[theoretical.length - 1] - theoretical[0]);
        const intercept = sorted[0] - slope * theoretical[0];
        
        const trace2 = {
            x: [minVal, maxVal],
            y: [slope * minVal + intercept, slope * maxVal + intercept],
            mode: 'lines',
            name: 'Reference Line',
            line: {
                color: '#dc3545',
                dash: 'dash',
                width: 2
            }
        };
        
        const layout = {
            title: 'Q-Q Plot',
            xaxis: { title: 'Theoretical Quantiles' },
            yaxis: { title: 'Sample Quantiles' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 60, r: 30, t: 50, b: 50 },
            showlegend: true
        };
        
        Plotly.newPlot('qqPlot', [trace, trace2], layout, { 
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
        
    } catch (error) {
        console.error('Error creating QQ plot:', error);
    }
}
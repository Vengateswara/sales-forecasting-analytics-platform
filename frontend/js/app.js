// frontend/js/app.js
// Main Application JavaScript

// ===== Global State =====
const appState = {
    datasetInfo: null,
    qualityReport: null,
    edaData: null,
    preprocessingResults: null,
    featureEngineering: null,
    modelResults: null,
    forecastResults: null,
    currentSection: 'upload',
    engineered_data: null,
    best_model: null
};

// ===== Dataset Preview Variables =====
let datasetData = [];
let filteredData = [];
let currentPage = 1;
let rowsPerPage = 10;
let tableHeaders = [];
let sortDirection = {};

// ===== Loading Screen =====
function showLoading(message = 'Processing...', progress = 0) {
    const overlay = document.getElementById('loadingScreen');
    const msgEl = document.getElementById('loadingMessage');
    const progressBar = document.getElementById('progressBar');
    
    if (overlay) {
        overlay.style.display = 'flex';
        if (msgEl) msgEl.textContent = message;
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
    }
}

function updateProgress(progress) {
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingScreen');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
    const statusDiv = document.getElementById('uploadStatus');
    if (!statusDiv) {
        const toastContainer = document.createElement('div');
        toastContainer.style.position = 'fixed';
        toastContainer.style.top = '80px';
        toastContainer.style.right = '20px';
        toastContainer.style.zIndex = '9999';
        toastContainer.style.maxWidth = '400px';
        document.body.appendChild(toastContainer);
        
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';
        
        toastContainer.innerHTML = `
            <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        setTimeout(() => {
            toastContainer.remove();
        }, 5000);
        return;
    }
    
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    statusDiv.innerHTML = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
}

// ===== Clear All Data =====
function clearAllData() {
    if (!confirm('Are you sure you want to clear all uploaded data and results?')) {
        return;
    }
    
    appState.datasetInfo = null;
    appState.qualityReport = null;
    appState.edaData = null;
    appState.preprocessingResults = null;
    appState.featureEngineering = null;
    appState.modelResults = null;
    appState.forecastResults = null;
    appState.engineered_data = null;
    appState.best_model = null;
    
    datasetData = [];
    filteredData = [];
    currentPage = 1;
    sortDirection = {};
    
    document.getElementById('datasetInfo').style.display = 'none';
    document.getElementById('uploadStatus').innerHTML = '';
    document.getElementById('qualityReport').innerHTML = '';
    document.getElementById('preprocessResults').innerHTML = '';
    document.getElementById('featureResults').innerHTML = '';
    document.getElementById('trainingResults').innerHTML = '';
    document.getElementById('evaluationResults').innerHTML = '';
    document.getElementById('comparisonResults').innerHTML = '';
    document.getElementById('forecastResults').innerHTML = '';
    document.getElementById('datasetPreview').style.display = 'none';
    
    const edaContainer = document.getElementById('edaCharts');
    if (edaContainer) {
        edaContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                Click <strong>"Generate EDA"</strong> to explore your data. Make sure you have completed <strong>Feature Engineering</strong> first.
            </div>
        `;
    }
    
    document.getElementById('fileInput').value = '';
    showToast('All data cleared successfully!', 'success');
    showSection('upload');
}

// ===== Section Navigation =====
function showSection(sectionName) {
    document.querySelectorAll('.section-content').forEach(el => {
        el.style.display = 'none';
    });
    
    const section = document.getElementById(`${sectionName}Section`);
    if (section) {
        section.style.display = 'block';
        appState.currentSection = sectionName;
    }
    
    document.querySelectorAll('.list-group-item').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.trim().toLowerCase().includes(sectionName)) {
            btn.classList.add('active');
            btn.style.backgroundColor = 'rgba(74, 144, 217, 0.3)';
            btn.style.borderLeft = '3px solid #4a90d9';
        } else {
            btn.style.backgroundColor = 'transparent';
            btn.style.borderLeft = 'none';
        }
    });
    
    const sidebarEl = document.getElementById('sidebar');
    if (sidebarEl) {
        const bsOffcanvas = bootstrap.Offcanvas.getInstance(sidebarEl);
        if (bsOffcanvas) {
            bsOffcanvas.hide();
        }
    }
    
    if (sectionName === 'eda') {
        loadEDACharts();
    } else if (sectionName === 'train') {
        if (appState.modelResults) {
            displayEvaluationResults();
        }
    } else if (sectionName === 'compare') {
        displayComparisonResults();
    } else if (sectionName === 'quality') {
        if (appState.qualityReport) {
            displayQualityReport(appState.qualityReport);
        }
    }
}

// ===== File Upload =====
document.addEventListener('DOMContentLoaded', function() {
    showSection('upload');
    
    const form = document.getElementById('uploadForm');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                showToast('Please select a file first', 'warning');
                return;
            }
            
            if (file.size > 100 * 1024 * 1024) {
                showToast('File size exceeds 100MB limit', 'error');
                return;
            }
            
            showLoading('Uploading Dataset...', 10);
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                updateProgress(30);
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Upload failed');
                }
                
                updateProgress(80);
                const result = await response.json();
                
                if (result.status === 'success') {
                    appState.datasetInfo = result.data;
                    appState.qualityReport = result.quality_report;
                    
                    displayDatasetInfo(result.data);
                    displayQualityReport(result.quality_report);
                    
                    showToast('File uploaded successfully!', 'success');
                    
                    const datasetInfo = document.getElementById('datasetInfo');
                    if (datasetInfo) datasetInfo.style.display = 'block';
                } else {
                    throw new Error(result.message || 'Upload failed');
                }
            } catch (error) {
                showToast(error.message, 'error');
                console.error('Upload error:', error);
            } finally {
                hideLoading();
            }
        });
    }
});

// ===== Display Dataset Info =====
function displayDatasetInfo(data) {
    const details = document.getElementById('datasetDetails');
    if (!details) return;
    
    details.innerHTML = `
        <div class="col-md-3 col-sm-6 mb-3">
            <div class="kpi-card">
                <div class="kpi-value">${data.rows.toLocaleString()}</div>
                <div class="kpi-label">Rows</div>
            </div>
        </div>
        <div class="col-md-3 col-sm-6 mb-3">
            <div class="kpi-card">
                <div class="kpi-value">${data.columns}</div>
                <div class="kpi-label">Columns</div>
            </div>
        </div>
        <div class="col-md-3 col-sm-6 mb-3">
            <div class="kpi-card">
                <div class="kpi-value">${(data.file_size / 1024 / 1024).toFixed(2)} MB</div>
                <div class="kpi-label">File Size</div>
            </div>
        </div>
        <div class="col-md-3 col-sm-6 mb-3">
            <div class="kpi-card">
                <div class="kpi-value">${data.memory_usage.toFixed(2)} MB</div>
                <div class="kpi-label">Memory Usage</div>
            </div>
        </div>
        <div class="col-12 mt-3">
            <div class="table-responsive">
                <table class="table table-sm table-hover">
                    <thead class="table-light">
                        <tr>
                            <th>Column Name</th>
                            <th>Data Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.column_names.map(col => `
                            <tr>
                                <td><code>${col}</code></td>
                                <td><span class="badge bg-secondary">${data.data_types[col] || 'Unknown'}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// ===== Display Quality Report =====
function displayQualityReport(report) {
    const container = document.getElementById('qualityReport');
    if (!container) return;
    
    const score = report.overall_quality_score || 0;
    let scoreColor = 'danger';
    let scoreLabel = 'Poor';
    if (score >= 80) { scoreColor = 'success'; scoreLabel = 'Excellent'; }
    else if (score >= 60) { scoreColor = 'warning'; scoreLabel = 'Good'; }
    else if (score >= 40) { scoreColor = 'warning'; scoreLabel = 'Fair'; }
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-3">
                <div class="kpi-card border-${scoreColor}">
                    <div class="kpi-value" style="color: var(--${scoreColor}-color);">${score}%</div>
                    <div class="kpi-label">Data Quality Score - ${scoreLabel}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card border-warning">
                    <div class="kpi-value">${report.missing_values ? Object.keys(report.missing_values).length : 0}</div>
                    <div class="kpi-label">Columns with Missing Values</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card border-danger">
                    <div class="kpi-value">${report.duplicate_rows || 0}</div>
                    <div class="kpi-label">Duplicate Rows</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card border-info">
                    <div class="kpi-value">${report.outliers ? Object.keys(report.outliers).length : 0}</div>
                    <div class="kpi-label">Columns with Outliers</div>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-white">
                        <h6 class="mb-0"><i class="fas fa-exclamation-triangle me-2 text-warning"></i>Missing Values</h6>
                    </div>
                    <div class="card-body">
                        ${report.missing_values && Object.keys(report.missing_values).length > 0 ? 
                            Object.entries(report.missing_values).map(([col, count]) => `
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <span><code>${col}</code></span>
                                    <span class="badge bg-warning">${count}</span>
                                </div>
                            `).join('') : 
                            '<p class="text-success mb-0"><i class="fas fa-check-circle me-2"></i>No missing values found</p>'
                        }
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-white">
                        <h6 class="mb-0"><i class="fas fa-chart-line me-2 text-danger"></i>Outliers</h6>
                    </div>
                    <div class="card-body">
                        ${report.outliers && Object.keys(report.outliers).length > 0 ?
                            Object.entries(report.outliers).map(([col, data]) => `
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <span><code>${col}</code></span>
                                    <span class="badge bg-danger">${data.count} (${data.percentage.toFixed(1)}%)</span>
                                </div>
                            `).join('') :
                            '<p class="text-success mb-0"><i class="fas fa-check-circle me-2"></i>No outliers detected</p>'
                        }
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ===== API Calls =====
async function apiCall(endpoint, method = 'POST', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`/api${endpoint}`, options);
    
    if (!response.ok) {
        let errorMessage = `API call failed: ${response.status}`;
        try {
            const error = await response.json();
            errorMessage = error.detail || errorMessage;
        } catch (e) {
            const text = await response.text();
            if (text) errorMessage = text;
        }
        throw new Error(errorMessage);
    }
    
    return await response.json();
}

// ===== Preprocess Data =====
async function preprocessData() {
    showLoading('Preprocessing Data...', 20);
    
    try {
        updateProgress(40);
        const result = await apiCall('/preprocess');
        
        updateProgress(80);
        if (result.status === 'success') {
            appState.preprocessingResults = result;
            displayPreprocessResults(result);
            showToast('Data preprocessing completed!', 'success');
        }
    } catch (error) {
        showToast(error.message, 'error');
        console.error('Preprocess error:', error);
    } finally {
        hideLoading();
    }
}

function displayPreprocessResults(result) {
    const container = document.getElementById('preprocessResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="alert alert-success mt-3">
            <i class="fas fa-check-circle me-2"></i>
            ${result.message}
        </div>
        <div class="row mt-3">
            <div class="col-md-4">
                <div class="kpi-card border-success">
                    <div class="kpi-value">${result.shape[0].toLocaleString()}</div>
                    <div class="kpi-label">Cleaned Rows</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="kpi-card border-primary">
                    <div class="kpi-value">${result.shape[1]}</div>
                    <div class="kpi-label">Columns</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="kpi-card border-info">
                    <div class="kpi-value">${result.preprocess_log ? result.preprocess_log.length : 0}</div>
                    <div class="kpi-label">Steps Performed</div>
                </div>
            </div>
        </div>
    `;
}

// ===== Feature Engineering =====
async function engineerFeatures() {
    showLoading('Engineering Features...', 20);
    
    try {
        updateProgress(40);
        const result = await apiCall('/feature-engineering');
        
        updateProgress(80);
        if (result.status === 'success') {
            appState.featureEngineering = result;
            appState.engineered_data = true;
            displayFeatureResults(result);
            showToast('Feature engineering completed!', 'success');
        }
    } catch (error) {
        showToast(error.message, 'error');
        console.error('Feature engineering error:', error);
    } finally {
        hideLoading();
    }
}

// ===== Display Feature Results with Dataset Preview =====
function displayFeatureResults(result) {
    const container = document.getElementById('featureResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="alert alert-success mt-3">
            <i class="fas fa-check-circle me-2"></i>
            ${result.message}
        </div>
        <div class="row mt-3">
            <div class="col-md-4">
                <div class="kpi-card border-success">
                    <div class="kpi-value">${result.feature_info.total_features}</div>
                    <div class="kpi-label">New Features</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="kpi-card border-primary">
                    <div class="kpi-value">${result.shape[1]}</div>
                    <div class="kpi-label">Total Features</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="kpi-card border-info">
                    <div class="kpi-value">${result.shape[0].toLocaleString()}</div>
                    <div class="kpi-label">Samples</div>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <button class="btn btn-outline-success btn-sm" onclick="downloadDataset()">
                <i class="fas fa-download me-1"></i>Download Dataset (CSV)
            </button>
            <button class="btn btn-outline-primary btn-sm" onclick="loadDatasetPreview()">
                <i class="fas fa-eye me-1"></i>Preview Dataset Schema
            </button>
        </div>
    `;
    
    setTimeout(() => {
        loadDatasetPreview();
    }, 500);
}

// ===== Load Dataset Preview (Show Column Info Only) =====
async function loadDatasetPreview() {
    try {
        showLoading('Loading dataset information...', 30);
        
        const response = await fetch('/api/get-engineered-data');
        
        if (!response.ok) {
            throw new Error('Failed to load dataset preview');
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            tableHeaders = result.headers;
            datasetData = result.data.slice(0, 5);
            
            const previewContainer = document.getElementById('datasetPreview');
            if (previewContainer) {
                previewContainer.style.display = 'block';
            }
            
            displayColumnInfo(result.headers, result.total_rows);
            
            showToast(`Dataset loaded: ${result.total_rows} rows, ${result.headers.length} columns`, 'success');
        }
    } catch (error) {
        console.error('Error loading dataset preview:', error);
        showToast('Error loading dataset preview: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// ===== Display Column Information =====
function displayColumnInfo(headers, totalRows) {
    const container = document.getElementById('datasetPreview');
    if (!container) return;
    
    // Define column categories
    const originalCols = ['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Total', 'Date', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating'];
    
    const engineeredCols = headers.filter(h => !originalCols.includes(h));
    
    // Category definitions
    const categories = {
        '📋 Original Columns': originalCols.filter(h => headers.includes(h)),
        '⏰ Time Features': ['year', 'quarter', 'month', 'day', 'day_of_week', 'week_of_year', 'weekend'],
        '🔄 Lag Features': ['lag_1', 'lag_7', 'lag_30'],
        '📊 Rolling Statistics': headers.filter(h => h.startsWith('rolling_')),
        '📈 Trend Features': ['growth_rate', 'difference', 'ema'],
        '🔄 Seasonality': ['month_sin', 'month_cos', 'dow_sin', 'dow_cos']
    };
    
    let html = `
        <div class="card mt-3">
            <div class="card-header bg-white d-flex justify-content-between align-items-center">
                <h6 class="mb-0"><i class="fas fa-table me-2"></i>Dataset Schema - ${totalRows} rows, ${headers.length} columns</h6>
                <span class="badge bg-primary">${engineeredCols.length} Engineered Features</span>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
                            <table class="table table-sm table-hover">
                                <thead class="table-light sticky-top">
                                    <tr>
                                        <th>#</th>
                                        <th>Column Name</th>
                                        <th>Type</th>
                                        <th>Category</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${headers.map((col, index) => {
                                        let category = 'Engineered';
                                        let type = 'Numeric';
                                        let badgeColor = 'bg-secondary';
                                        
                                        if (originalCols.includes(col)) {
                                            category = 'Original';
                                            type = col === 'Date' || col === 'Time' ? 'DateTime' : 
                                                   ['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment'].includes(col) ? 'Categorical' : 
                                                   'Numeric';
                                            badgeColor = 'bg-success';
                                        } else if (col.startsWith('rolling_')) {
                                            category = 'Rolling Stat';
                                            badgeColor = 'bg-warning text-dark';
                                        } else if (['year', 'quarter', 'month', 'day', 'day_of_week', 'week_of_year', 'weekend'].includes(col)) {
                                            category = 'Time Feature';
                                            badgeColor = 'bg-primary';
                                        } else if (col.startsWith('lag_')) {
                                            category = 'Lag Feature';
                                            badgeColor = 'bg-info';
                                        } else if (['growth_rate', 'difference', 'ema'].includes(col)) {
                                            category = 'Trend Feature';
                                            badgeColor = 'bg-danger';
                                        } else if (['month_sin', 'month_cos', 'dow_sin', 'dow_cos'].includes(col)) {
                                            category = 'Seasonality';
                                            badgeColor = 'bg-purple';
                                        }
                                        
                                        let typeBadge = 'bg-secondary';
                                        if (type === 'Categorical') typeBadge = 'bg-info';
                                        else if (type === 'DateTime') typeBadge = 'bg-primary';
                                        else if (type === 'Numeric') typeBadge = 'bg-success';
                                        
                                        return `
                                            <tr>
                                                <td>${index + 1}</td>
                                                <td><code>${col}</code></td>
                                                <td><span class="badge ${typeBadge}">${type}</span></td>
                                                <td><span class="badge ${badgeColor}">${category}</span></td>
                                            </tr>
                                        `;
                                    }).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-white">
                                <h6 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Column Summary</h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6">
                                        <div class="kpi-card border-success">
                                            <div class="kpi-value" style="font-size: 1.2rem;">${headers.length}</div>
                                            <div class="kpi-label">Total Columns</div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="kpi-card border-primary">
                                            <div class="kpi-value" style="font-size: 1.2rem;">${originalCols.filter(h => headers.includes(h)).length}</div>
                                            <div class="kpi-label">Original Columns</div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="kpi-card border-info">
                                            <div class="kpi-value" style="font-size: 1.2rem;">${engineeredCols.length}</div>
                                            <div class="kpi-label">Engineered Features</div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="kpi-card border-warning">
                                            <div class="kpi-value" style="font-size: 1.2rem;">${totalRows}</div>
                                            <div class="kpi-label">Total Rows</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <hr>
                                
                                <h6 class="mt-2">Feature Categories</h6>
                                ${Object.entries(categories).map(([category, cols]) => {
                                    const filtered = cols.filter(c => headers.includes(c));
                                    if (filtered.length === 0) return '';
                                    return `
                                        <div class="d-flex justify-content-between align-items-center mb-1">
                                            <span style="font-size: 0.9rem;">${category}</span>
                                            <span class="badge bg-secondary">${filtered.length}</span>
                                        </div>
                                    `;
                                }).join('')}
                                
                                <hr>
                                
                                <div class="mt-2">
                                    <h6 class="mb-2">Feature Counts</h6>
                                    <div class="progress mb-2" style="height: 20px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: ${(originalCols.filter(h => headers.includes(h)).length / headers.length * 100)}%" aria-valuenow="${(originalCols.filter(h => headers.includes(h)).length / headers.length * 100)}" aria-valuemin="0" aria-valuemax="100">
                                            Original
                                        </div>
                                        <div class="progress-bar bg-primary" role="progressbar" style="width: ${(engineeredCols.length / headers.length * 100)}%" aria-valuenow="${(engineeredCols.length / headers.length * 100)}" aria-valuemin="0" aria-valuemax="100">
                                            Engineered
                                        </div>
                                    </div>
                                    <div class="d-flex justify-content-between small">
                                        <span>${(originalCols.filter(h => headers.includes(h)).length / headers.length * 100).toFixed(0)}% Original</span>
                                        <span>${(engineeredCols.length / headers.length * 100).toFixed(0)}% Engineered</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ===== Download Dataset =====
function downloadDataset() {
    if (!datasetData || datasetData.length === 0) {
        showToast('No data to download', 'warning');
        return;
    }
    
    const headers = tableHeaders.join(',');
    const rows = datasetData.map(row => 
        tableHeaders.map(header => {
            const val = row[header] !== undefined ? row[header] : '';
            return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
        }).join(',')
    );
    
    const csv = [headers, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'engineered_dataset.csv';
    a.click();
    window.URL.revokeObjectURL(url);
    
    showToast('Dataset downloaded successfully!', 'success');
}

// ===== Train Models =====
async function trainModels() {
    showLoading('Training Models... This may take several minutes...', 10);
    
    try {
        updateProgress(30);
        const result = await apiCall('/train');
        
        updateProgress(80);
        if (result.status === 'success') {
            appState.modelResults = result.results;
            appState.best_model = {
                name: result.best_model,
                testing_score: result.metrics.test_r2,
                cv_score: result.metrics.cv_score || 0,
                metrics: result.metrics,
                training_score: result.metrics.train_r2 || 0
            };
            
            displayTrainingResults(result);
            displayEvaluationResults();
            
            showToast(`Model training completed! Best model: ${result.best_model}`, 'success');
        }
    } catch (error) {
        showToast(error.message, 'error');
        console.error('Training error:', error);
    } finally {
        hideLoading();
    }
}

// ===== Display Training Results =====
function displayTrainingResults(result) {
    const container = document.getElementById('trainingResults');
    if (!container) return;
    
    const bestModelName = result.best_model;
    const sortedModels = Object.entries(result.results)
        .sort((a, b) => b[1].testing_score - a[1].testing_score);
    
    const getRankBadge = (index) => {
        if (index === 0) return '🥇';
        if (index === 1) return '🥈';
        if (index === 2) return '🥉';
        return `#${index + 1}`;
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
                    <div class="kpi-label">🏆 Best Model</div>
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
        
        <div class="card mt-3">
            <div class="card-header bg-white d-flex justify-content-between align-items-center">
                <h6 class="mb-0"><i class="fas fa-ranking-star me-2"></i>Model Rankings (Best to Worst)</h6>
                <span class="badge bg-secondary">${sortedModels.length} models</span>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-hover mb-0">
                        <thead class="table-light">
                            <tr>
                                <th>Rank</th>
                                <th>Model</th>
                                <th>Testing R²</th>
                                <th>Training R²</th>
                                <th>CV Score</th>
                                <th>MAE</th>
                                <th>RMSE</th>
                                <th>Train Time (s)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${sortedModels.map(([name, data], index) => `
                                <tr class="${name === bestModelName ? 'table-success' : ''}">
                                    <td>
                                        <span class="badge ${index === 0 ? 'bg-success' : index === 1 ? 'bg-primary' : index === 2 ? 'bg-info' : 'bg-secondary'}">
                                            ${getRankBadge(index)}
                                        </span>
                                    </td>
                                    <td>
                                        <strong>${name}</strong> 
                                        ${name === bestModelName ? '🏆' : ''}
                                    </td>
                                    <td>${(data.testing_score * 100).toFixed(2)}%</td>
                                    <td>${(data.training_score * 100).toFixed(2)}%</td>
                                    <td>${data.cv_score ? data.cv_score.toFixed(4) : 'N/A'}</td>
                                    <td>${data.metrics.test_mae.toFixed(2)}</td>
                                    <td>${data.metrics.test_rmse.toFixed(2)}</td>
                                    <td>${data.train_time.toFixed(2)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="text-center mt-2">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                Models sorted by Testing R² (best to worst). 🏆 = Best overall model.
            </small>
        </div>
        <hr class="my-3">
    `;
}

// ===== Display Evaluation Results =====
function displayEvaluationResults() {
    const container = document.getElementById('evaluationResults');
    if (!container) return;
    
    if (!appState.modelResults) {
        container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No model evaluation results available. Please train models first.
            </div>
        `;
        container.style.display = 'block';
        return;
    }
    
    const results = appState.modelResults;
    const bestModel = appState.best_model;
    const sortedModels = Object.entries(results)
        .sort((a, b) => b[1].testing_score - a[1].testing_score);
    
    container.style.display = 'block';
    
    const getRankLabel = (index) => {
        if (index === 0) return '🥇 Best';
        if (index === 1) return '🥈 Second';
        if (index === 2) return '🥉 Third';
        return `#${index + 1}`;
    };
    
    const getRankBadge = (index) => {
        if (index === 0) return 'bg-success';
        if (index === 1) return 'bg-primary';
        if (index === 2) return 'bg-info';
        return 'bg-secondary';
    };
    
    let html = `
        <h5 class="mb-3"><i class="fas fa-chart-bar me-2"></i>Detailed Model Evaluation</h5>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-white">
                        <h6 class="mb-0"><i class="fas fa-trophy me-2 text-warning"></i>🏆 Best Model: ${bestModel ? bestModel.name : 'N/A'}</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="kpi-card border-success">
                                    <div class="kpi-value">${bestModel ? (bestModel.testing_score * 100).toFixed(2) : 'N/A'}%</div>
                                    <div class="kpi-label">R² Score</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="kpi-card border-primary">
                                    <div class="kpi-value">${bestModel ? bestModel.metrics.test_mae.toFixed(2) : 'N/A'}</div>
                                    <div class="kpi-label">MAE</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="kpi-card border-warning">
                                    <div class="kpi-value">${bestModel ? bestModel.metrics.test_rmse.toFixed(2) : 'N/A'}</div>
                                    <div class="kpi-label">RMSE</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="kpi-card border-info">
                                    <div class="kpi-value">${bestModel ? bestModel.cv_score.toFixed(4) : 'N/A'}</div>
                                    <div class="kpi-label">CV Score</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-white d-flex justify-content-between align-items-center">
                        <h6 class="mb-0"><i class="fas fa-ranking-star me-2"></i>All Models (Ranked Best to Worst)</h6>
                        <span class="badge bg-secondary">${sortedModels.length} models</span>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table table-hover mb-0">
                                <thead class="table-light">
                                    <tr>
                                        <th style="width: 60px;">Rank</th>
                                        <th>Model</th>
                                        <th>Training R²</th>
                                        <th>Testing R²</th>
                                        <th>CV Score</th>
                                        <th>MAE</th>
                                        <th>RMSE</th>
                                        <th>MAPE (%)</th>
                                        <th>Train Time (s)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${sortedModels.map(([name, data], index) => `
                                        <tr class="${name === bestModel?.name ? 'table-success' : ''}">
                                            <td>
                                                <span class="badge ${getRankBadge(index)}" style="font-size: 0.9rem;">
                                                    ${getRankLabel(index)}
                                                </span>
                                            </td>
                                            <td>
                                                <strong>${name}</strong> 
                                                ${name === bestModel?.name ? '🏆' : ''}
                                            </td>
                                            <td>${(data.training_score * 100).toFixed(2)}%</td>
                                            <td><strong>${(data.testing_score * 100).toFixed(2)}%</strong></td>
                                            <td>${data.cv_score ? data.cv_score.toFixed(4) : 'N/A'}</td>
                                            <td>${data.metrics.test_mae.toFixed(2)}</td>
                                            <td>${data.metrics.test_rmse.toFixed(2)}</td>
                                            <td>${data.metrics.test_mape ? data.metrics.test_mape.toFixed(2) : 'N/A'}</td>
                                            <td>${data.train_time.toFixed(2)}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h6 class="text-muted">🥇 Best Performing Model</h6>
                        <h3 class="text-success">${sortedModels[0]?.[0] || 'N/A'}</h3>
                        <small class="text-muted">Testing R²: ${sortedModels[0] ? (sortedModels[0][1].testing_score * 100).toFixed(2) : 'N/A'}%</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h6 class="text-muted">⚡ Fastest Model</h6>
                        <h3 class="text-primary">${sortedModels.reduce((a, b) => a[1].train_time < b[1].train_time ? a : b)?.[0] || 'N/A'}</h3>
                        <small class="text-muted">Train Time: ${sortedModels.reduce((a, b) => a[1].train_time < b[1].train_time ? a : b)?.[1]?.train_time?.toFixed(3) || 'N/A'}s</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h6 class="text-muted">📊 Best CV Score</h6>
                        <h3 class="text-info">${sortedModels.reduce((a, b) => a[1].cv_score > b[1].cv_score ? a : b)?.[0] || 'N/A'}</h3>
                        <small class="text-muted">CV Score: ${sortedModels.reduce((a, b) => a[1].cv_score > b[1].cv_score ? a : b)?.[1]?.cv_score?.toFixed(4) || 'N/A'}</small>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ===== Display Comparison Results =====
function displayComparisonResults() {
    const container = document.getElementById('comparisonResults');
    if (!container) return;
    
    if (!appState.modelResults) {
        container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No model comparison data available. Please train models first.
            </div>
        `;
        return;
    }
    
    const results = appState.modelResults;
    const bestModel = appState.best_model;
    const bestName = bestModel ? bestModel.name : Object.keys(results)[0];
    
    let html = `
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-white">
                        <h6 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Model Performance Comparison</h6>
                    </div>
                    <div class="card-body">
                        <div id="comparisonChart"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    html += `
        <div class="row mt-3">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Best Performing Model</h6>
                        <h3 class="text-success">${bestName}</h3>
                        <small class="text-muted">Highest Combined Score</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Best R² Score</h6>
                        <h3 class="text-primary">${(Math.max(...Object.values(results).map(d => d.testing_score)) * 100).toFixed(2)}%</h3>
                        <small class="text-muted">${Object.keys(results).find(k => results[k].testing_score === Math.max(...Object.values(results).map(d => d.testing_score)))}</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Lowest MAE</h6>
                        <h3 class="text-info">${Math.min(...Object.values(results).map(d => d.metrics.test_mae)).toFixed(2)}</h3>
                        <small class="text-muted">${Object.keys(results).find(k => results[k].metrics.test_mae === Math.min(...Object.values(results).map(d => d.metrics.test_mae)))}</small>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    
    if (typeof createComparisonChart === 'function' && results) {
        setTimeout(() => {
            createComparisonChart(results);
        }, 200);
    }
}

// ===== Generate Forecast =====
async function generateForecast() {
    showLoading('Generating Forecast...', 20);
    
    try {
        const period = document.getElementById('forecastPeriod').value;
        
        updateProgress(40);
        const result = await apiCall(`/forecast?period=${period}`);
        
        updateProgress(80);
        if (result.status === 'success') {
            appState.forecastResults = result.forecast;
            displayForecast(result.forecast);
            showToast(`Forecast generated for ${period} days!`, 'success');
        }
    } catch (error) {
        showToast(error.message, 'error');
        console.error('Forecast error:', error);
    } finally {
        hideLoading();
    }
}

function displayForecast(forecast) {
    const container = document.getElementById('forecastResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row mt-3">
            <div class="col-md-3">
                <div class="kpi-card border-success">
                    <div class="kpi-value" style="font-size: 1rem;">${forecast.trend}</div>
                    <div class="kpi-label">Trend</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card border-primary">
                    <div class="kpi-value ${forecast.growth_rate >= 0 ? 'text-success' : 'text-danger'}">
                        ${forecast.growth_rate.toFixed(2)}%
                    </div>
                    <div class="kpi-label">Growth Rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card border-info">
                    <div class="kpi-value">${forecast.summary.mean.toFixed(2)}</div>
                    <div class="kpi-label">Average Forecast</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card border-warning">
                    <div class="kpi-value">${forecast.summary.std.toFixed(2)}</div>
                    <div class="kpi-label">Std Deviation</div>
                </div>
            </div>
        </div>
        <div class="card mt-3">
            <div class="card-header bg-white">
                <h6 class="mb-0"><i class="fas fa-chart-line me-2"></i>Forecast Chart</h6>
            </div>
            <div class="card-body">
                <div id="forecastChart"></div>
            </div>
        </div>
    `;
    
    if (typeof createForecastChart === 'function') {
        setTimeout(() => createForecastChart(forecast), 100);
    }
}

// ===== Download File =====
function downloadFile(fileType) {
    window.location.href = `/api/download/${fileType}`;
}

// Global error handler
window.onerror = function(message, source, lineno, colno, error) {
    console.error('Global error:', message, error);
    hideLoading();
    showToast('An unexpected error occurred. Please check the console for details.', 'error');
    return true;
};
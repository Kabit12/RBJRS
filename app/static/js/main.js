/**
 * RBJRS Main JavaScript
 * ======================
 * Core interactivity for the application.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages after 5 seconds
    const flashAlerts = document.querySelectorAll('.toast-container .alert');
    flashAlerts.forEach(function(alert) {
        setTimeout(function() {
            alert.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            alert.style.opacity = '0';
            alert.style.transform = 'translateX(20px)';
            setTimeout(function() { alert.remove(); }, 300);
        }, 5000);
    });

    // Mobile sidebar toggle
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });
    }

    // File upload drag-and-drop zones
    const uploadZones = document.querySelectorAll('.file-upload-zone');
    uploadZones.forEach(function(zone) {
        const fileInput = zone.querySelector('input[type="file"]');

        zone.addEventListener('dragover', function(e) {
            e.preventDefault();
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', function() {
            zone.classList.remove('drag-over');
        });

        zone.addEventListener('drop', function(e) {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (fileInput && e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileDisplay(zone, e.dataTransfer.files[0]);
            }
        });

        zone.addEventListener('click', function() {
            if (fileInput) fileInput.click();
        });

        if (fileInput) {
            fileInput.addEventListener('change', function() {
                if (this.files.length) {
                    updateFileDisplay(zone, this.files[0]);
                }
            });
        }
    });

    // Modal close on backdrop click
    document.querySelectorAll('.modal-backdrop').forEach(function(backdrop) {
        backdrop.addEventListener('click', function(e) {
            if (e.target === backdrop) {
                backdrop.classList.remove('active');
            }
        });
    });
});

function updateFileDisplay(zone, file) {
    const textEl = zone.querySelector('.upload-text');
    if (textEl) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        textEl.innerHTML = '<strong>' + file.name + '</strong><br>' + sizeMB + ' MB';
    }
}

function openModal(id) {
    document.getElementById(id).classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

// Confirmation dialog
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

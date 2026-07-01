document.addEventListener("DOMContentLoaded", function() {
    // 1. Sidebar Panel Toggling Logic
    const sidebarLinks = document.querySelectorAll(".sidebar-link");
    const panelContents = document.querySelectorAll(".panel-content");
    const pageTitle = document.getElementById("page-title");
    
    const titleMappings = {
        "apply-section": "Credit Card Application Portal",
        "model-section": "Model Tuning & Performance",
        "dataset-section": "Dataset Distributions & Insights"
    };

    sidebarLinks.forEach(link => {
        link.addEventListener("click", function() {
            // Remove active status from all nav buttons
            sidebarLinks.forEach(btn => btn.classList.remove("active"));
            
            // Add active status to clicked nav button
            this.classList.add("active");
            
            // Hide all panel content sections
            panelContents.forEach(panel => panel.classList.add("d-none"));
            
            // Get target panel ID and display it
            const targetId = this.getAttribute("data-target");
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.remove("d-none");
            }
            
            // Dynamically update main heading title
            if (pageTitle && titleMappings[targetId]) {
                pageTitle.textContent = titleMappings[targetId];
            }
        });
    });

    // 2. Form Input Fields Toggling Logic
    const employmentStatusSelect = document.getElementById("employment_status");
    const yearsEmployedGroup = document.getElementById("years_employed_group");
    const yearsEmployedInput = document.getElementById("years_employed");
    const predictionForm = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");
    
    // Toggle Years Employed input based on Employment Status
    function toggleEmploymentFields() {
        if (!employmentStatusSelect || !yearsEmployedGroup || !yearsEmployedInput) return;
        
        if (employmentStatusSelect.value === "unemployed") {
            yearsEmployedGroup.classList.add("d-none");
            yearsEmployedInput.value = "0";
            yearsEmployedInput.removeAttribute("required");
        } else {
            yearsEmployedGroup.classList.remove("d-none");
            yearsEmployedInput.setAttribute("required", "required");
        }
    }
    
    if (employmentStatusSelect) {
        employmentStatusSelect.addEventListener("change", toggleEmploymentFields);
        toggleEmploymentFields(); // Initial run on page load
    }
    
    // 3. Client-Side Input Form Validation
    if (predictionForm) {
        predictionForm.addEventListener("submit", function(event) {
            // Read values for validation
            const income = parseFloat(document.getElementById("income").value);
            const age = parseFloat(document.getElementById("age").value);
            const children = parseInt(document.getElementById("children").value);
            const familyMembers = parseFloat(document.getElementById("family_members").value);
            const yearsEmployed = parseFloat(yearsEmployedInput.value);
            
            let isValid = true;
            let errorMessage = "";
            
            // Age validation
            if (isNaN(age) || age < 18 || age > 100) {
                isValid = false;
                errorMessage += "- Age must be between 18 and 100 years.\n";
            }
            
            // Income validation
            if (isNaN(income) || income < 0) {
                isValid = false;
                errorMessage += "- Annual income cannot be negative.\n";
            }
            
            // Employment duration validation
            if (employmentStatusSelect.value === "employed" && (isNaN(yearsEmployed) || yearsEmployed < 0 || yearsEmployed > age - 15)) {
                isValid = false;
                errorMessage += "- Employment years must be positive and reasonable for your age.\n";
            }
            
            // Children/Family members validation
            if (isNaN(children) || children < 0) {
                isValid = false;
                errorMessage += "- Number of children cannot be negative.\n";
            }
            
            if (isNaN(familyMembers) || familyMembers < 1) {
                isValid = false;
                errorMessage += "- Total family members count must be at least 1.\n";
            }
            
            if (familyMembers < children + 1) {
                isValid = false;
                errorMessage += "- Total family members must be at least children count + 1 (applicant).\n";
            }
            
            if (!isValid) {
                event.preventDefault();
                alert("Validation Error:\n" + errorMessage);
                return;
            }
            
            // Add loading spinner state to button
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Evaluating Application...';
        });
    }
});

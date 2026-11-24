/* ==========================================================================
   Diabetes Prediction System — Vanilla JavaScript Controller
   Supports both index.html (Landing) and prediction.html (Assessment)
   Runs real browser-side Random Forest predict_proba() + Platt calibration
   using window.DIABETES_RF_MODEL (exported from dataset Update - new.csv).
   ========================================================================== */

(function () {
  'use strict';

  var TOTAL_QUESTIONS = 9;
  var TOTAL_STEPS = 3;

  var STEP_CONFIG = {
    1: {
      title: 'Step 1 — Profile & Weekly Exercise',
      fields: ['ageGroup', 'exerciseDays']
    },
    2: {
      title: 'Step 2 — Family History, Blood Pressure & Cravings',
      fields: ['familyHistory', 'highBloodPressure', 'craveSweets']
    },
    3: {
      title: 'Step 3 — Health Symptoms & Recent Changes',
      fields: ['frequentUrination', 'excessiveThirst', 'frequentHunger', 'weightChange']
    }
  };

  var ALL_FIELDS = [
    'ageGroup',
    'exerciseDays',
    'familyHistory',
    'highBloodPressure',
    'craveSweets',
    'frequentUrination',
    'excessiveThirst',
    'frequentHunger',
    'weightChange'
  ];

  var FIELD_LABELS = {
    ageGroup: 'Age Group',
    exerciseDays: 'Weekly Exercise',
    familyHistory: 'Family History of Diabetes',
    highBloodPressure: 'High Blood Pressure History',
    craveSweets: 'Sweet Cravings',
    frequentUrination: 'Frequent Urination',
    excessiveThirst: 'Excessive Thirst',
    frequentHunger: 'Frequent Hunger',
    weightChange: 'Weight Change (Last 3 Months)'
  };

  var state = {
    currentStep: 1,
    answers: {}
  };

  /* Shared Navigation DOM */
  var header = document.getElementById('siteHeader');
  var menuToggle = document.getElementById('menuToggle');
  var mobileNav = document.getElementById('mobileNav');
  var mobileLinks = document.querySelectorAll('[data-mobile-link]');
  var navLinks = document.querySelectorAll('[data-nav-link]');

  /* Assessment DOM (only present on prediction.html) */
  var assessmentShell = document.getElementById('assessmentShell');
  var assessmentForm = document.getElementById('assessmentForm');
  var stepElements = document.querySelectorAll('.form-step');
  var stepIndicators = document.querySelectorAll('[data-step-indicator]');
  var stepCounter = document.getElementById('stepCounter');
  var stepHeading = document.getElementById('stepHeading');
  var answeredCount = document.getElementById('answeredCount');
  var progressPercent = document.getElementById('progressPercent');
  var progressBarTrack = document.getElementById('progressBarTrack');
  var progressBarFill = document.getElementById('progressBarFill');
  var validationBanner = document.getElementById('validationBanner');
  var validationMessage = document.getElementById('validationMessage');

  var prevBtn = document.getElementById('prevBtn');
  var nextBtn = document.getElementById('nextBtn');
  var submitBtn = document.getElementById('submitBtn');

  /* Result Dashboard DOM */
  var resultSection = document.getElementById('resultSection');
  var resultHeroCard = document.getElementById('resultHeroCard');
  var resultStatusIcon = document.getElementById('resultStatusIcon');
  var resultCategoryTitle = document.getElementById('resultCategoryTitle');
  var resultCategoryDesc = document.getElementById('resultCategoryDesc');
  var confidencePanel = document.getElementById('confidencePanel');
  var confidencePercentage = document.getElementById('confidencePercentage');
  var confidenceCategoryLabel = document.getElementById('confidenceCategoryLabel');
  var confidenceTreesMeta = document.getElementById('confidenceTreesMeta');
  var confidenceBarTrack = document.getElementById('confidenceBarTrack');
  var confidenceBarFill = document.getElementById('confidenceBarFill');
  var probYesVal = document.getElementById('probYesVal');
  var rawProbYesVal = document.getElementById('rawProbYesVal');
  var confidenceNote = document.getElementById('confidenceNote');
  var spectrumReadout = document.getElementById('spectrumReadout');
  var resultScaleMarker = document.getElementById('resultScaleMarker');
  var resultCardsGrid = document.getElementById('resultCardsGrid');
  var insightsList = document.getElementById('insightsList');
  var modelMetricsText = document.getElementById('modelMetricsText');

  var retakeBtn = document.getElementById('retakeBtn');
  var editAnswersBtn = document.getElementById('editAnswersBtn');

  /* ========================================================================
     1. Navigation, Mobile Menu & Smooth Scrolling
     ======================================================================== */
  function setMobileMenuOpen(isOpen) {
    if (!menuToggle || !mobileNav) return;
    menuToggle.setAttribute('aria-expanded', String(isOpen));
    mobileNav.setAttribute('aria-hidden', String(!isOpen));
    mobileNav.classList.toggle('is-open', isOpen);
  }

  function initNavigation() {
    if (menuToggle) {
      menuToggle.addEventListener('click', function () {
        var expanded = menuToggle.getAttribute('aria-expanded') === 'true';
        setMobileMenuOpen(!expanded);
      });
    }

    mobileLinks.forEach(function (link) {
      link.addEventListener('click', function () {
        setMobileMenuOpen(false);
      });
    });

    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
      anchor.addEventListener('click', function (event) {
        var targetId = anchor.getAttribute('href');
        if (!targetId || targetId === '#') return;
        var targetEl = document.querySelector(targetId);
        if (!targetEl) return;

        event.preventDefault();
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setMobileMenuOpen(false);
      });
    });

    window.addEventListener('scroll', handleScrollEffects, { passive: true });
    handleScrollEffects();
  }

  function handleScrollEffects() {
    if (header) {
      header.classList.toggle('is-scrolled', window.scrollY > 12);
    }

    if (navLinks.length === 0) return;

    var sections = ['home', 'about', 'features', 'disclaimer'];
    var scrollPos = window.scrollY + 140;
    var activeId = 'home';

    for (var i = 0; i < sections.length; i++) {
      var sectionEl = document.getElementById(sections[i]);
      if (sectionEl && sectionEl.offsetTop <= scrollPos) {
        activeId = sections[i];
      }
    }

    navLinks.forEach(function (link) {
      var matches = link.getAttribute('href') === '#' + activeId;
      link.classList.toggle('active', matches);
    });

    mobileLinks.forEach(function (link) {
      if (link.classList.contains('btn')) return;
      var href = link.getAttribute('href');
      if (!href || href.charAt(0) !== '#') return;
      link.classList.toggle('active', href === '#' + activeId);
    });
  }

  /* ========================================================================
     2. Scroll Reveal Animations
     ======================================================================== */
  function initScrollReveal() {
    var revealNodes = document.querySelectorAll('.reveal-on-scroll');
    if (revealNodes.length === 0) return;

    if (!('IntersectionObserver' in window)) {
      revealNodes.forEach(function (el) {
        el.classList.add('is-visible');
      });
      return;
    }

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -36px 0px' }
    );

    revealNodes.forEach(function (el) {
      if (!el.classList.contains('is-visible')) {
        observer.observe(el);
      }
    });
  }

  /* ========================================================================
     3. Questionnaire State, Validation & Directional Step Navigation
     ======================================================================== */
  function countAnsweredQuestions() {
    return ALL_FIELDS.filter(function (key) {
      return Boolean(state.answers[key]);
    }).length;
  }

  function updateProgressUI() {
    var step = state.currentStep;
    var pct = Math.round((step / TOTAL_STEPS) * 100);
    var answered = countAnsweredQuestions();

    if (stepCounter) stepCounter.textContent = 'Step ' + step + ' of ' + TOTAL_STEPS;
    if (stepHeading) stepHeading.textContent = STEP_CONFIG[step].title;
    if (answeredCount) answeredCount.textContent = answered + ' of ' + TOTAL_QUESTIONS + ' answered';
    if (progressPercent) progressPercent.textContent = pct + '%';
    if (progressBarFill) progressBarFill.style.width = pct + '%';
    if (progressBarTrack) progressBarTrack.setAttribute('aria-valuenow', String(pct));

    stepIndicators.forEach(function (indicator) {
      var indicatorStep = Number(indicator.getAttribute('data-step-indicator'));
      indicator.classList.remove('active', 'completed');
      if (indicatorStep < step) {
        indicator.classList.add('completed');
      } else if (indicatorStep === step) {
        indicator.classList.add('active');
      }
    });
  }

  function renderStep(stepNumber, direction) {
    state.currentStep = stepNumber;

    stepElements.forEach(function (stepEl) {
      var elStep = Number(stepEl.getAttribute('data-step'));
      var isCurrent = elStep === stepNumber;
      stepEl.hidden = !isCurrent;
      stepEl.classList.remove('active', 'anim-forward', 'anim-backward');
      if (isCurrent) {
        stepEl.classList.add('active');
        if (direction === 'backward') {
          stepEl.classList.add('anim-backward');
        } else {
          stepEl.classList.add('anim-forward');
        }
      }
    });

    if (prevBtn) prevBtn.hidden = stepNumber === 1;
    if (nextBtn) nextBtn.hidden = stepNumber === TOTAL_STEPS;
    if (submitBtn) submitBtn.hidden = stepNumber !== TOTAL_STEPS;

    hideValidationError();
    updateProgressUI();
  }

  function getMissingFieldsForStep(stepNumber) {
    var fields = STEP_CONFIG[stepNumber].fields;
    return fields.filter(function (fieldName) {
      return !state.answers[fieldName];
    });
  }

  function highlightMissingQuestions(missingFields) {
    document.querySelectorAll('.question-card').forEach(function (card) {
      var qName = card.getAttribute('data-question');
      card.classList.toggle('is-missing', missingFields.indexOf(qName) !== -1);
    });
  }

  function showCustomValidationError(message, missingFields) {
    highlightMissingQuestions(missingFields || []);
    if (!validationBanner || !validationMessage) return;

    validationMessage.textContent = message;
    validationBanner.hidden = false;

    if (missingFields && missingFields.length > 0) {
      var firstMissingCard = document.querySelector(
        '.question-card[data-question="' + missingFields[0] + '"]'
      );
      if (firstMissingCard) {
        firstMissingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }
    validationBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function showValidationError(missingFields) {
    var names = missingFields.map(function (field) {
      return FIELD_LABELS[field] || field;
    });
    showCustomValidationError(
      'Please complete all questions in this step before proceeding (' + names.join(', ') + ').',
      missingFields
    );
  }

  function hideValidationError() {
    if (validationBanner) {
      validationBanner.hidden = true;
    }
    highlightMissingQuestions([]);
  }

  function validateStep(stepNumber) {
    var missing = getMissingFieldsForStep(stepNumber);
    if (missing.length > 0) {
      showValidationError(missing);
      return false;
    }
    hideValidationError();
    return true;
  }

  function validateAllAnswersAgainstModel(answers, model) {
    var missingAll = ALL_FIELDS.filter(function (fieldName) {
      return !answers[fieldName];
    });
    if (missingAll.length > 0) {
      return {
        valid: false,
        missingFields: missingAll,
        message: 'Please answer all 9 required questions before generating a prediction.'
      };
    }

    var allowedByKey = {};
    for (var i = 0; i < model.features.length; i++) {
      var feat = model.features[i];
      allowedByKey[feat.key] = feat.categories;
    }

    var invalidFields = [];
    for (var j = 0; j < ALL_FIELDS.length; j++) {
      var key = ALL_FIELDS[j];
      var val = answers[key];
      var allowed = allowedByKey[key];
      if (!allowed || allowed.indexOf(val) === -1) {
        invalidFields.push(key);
      }
    }

    if (invalidFields.length > 0) {
      var labels = invalidFields.map(function (f) {
        return FIELD_LABELS[f] || f;
      });
      return {
        valid: false,
        missingFields: invalidFields,
        message:
          'Unexpected input value detected for: ' +
          labels.join(', ') +
          '. Please choose one of the supported options.'
      };
    }

    return { valid: true };
  }

  function syncAnswerFromInput(input) {
    if (!input || !input.name) return;
    state.answers[input.name] = input.value;

    var card = input.closest('.question-card');
    if (card) {
      card.classList.remove('is-missing');
    }

    if (validationBanner && !validationBanner.hidden) {
      var remainingMissing = getMissingFieldsForStep(state.currentStep);
      if (remainingMissing.length === 0) {
        hideValidationError();
      }
    }

    updateProgressUI();
  }

  function initQuestionnaire() {
    if (!assessmentForm) return;

    var radioInputs = assessmentForm.querySelectorAll('input[type="radio"]');
    radioInputs.forEach(function (radio) {
      if (radio.checked) {
        state.answers[radio.name] = radio.value;
      }
      radio.addEventListener('change', function () {
        syncAnswerFromInput(radio);
      });
    });

    if (prevBtn) {
      prevBtn.addEventListener('click', function () {
        if (state.currentStep > 1) {
          renderStep(state.currentStep - 1, 'backward');
          assessmentShell.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', function () {
        if (validateStep(state.currentStep) && state.currentStep < TOTAL_STEPS) {
          renderStep(state.currentStep + 1, 'forward');
          assessmentShell.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    }

    assessmentForm.addEventListener('submit', function (event) {
      event.preventDefault();
      if (!validateStep(state.currentStep)) {
        return;
      }
      handleAssessmentSubmission();
    });

    renderStep(1, 'forward');
  }

  /* ========================================================================
     4. Real Random Forest Inference (predict_proba + 5-fold Platt Calibration)
     ======================================================================== */
  function safeLogit(p) {
    var clamped = Math.max(1e-4, Math.min(1 - 1e-4, p));
    return Math.log(clamped / (1 - clamped));
  }

  function sigmoid(z) {
    if (z >= 0) {
      return 1 / (1 + Math.exp(-z));
    }
    var ez = Math.exp(z);
    return ez / (1 + ez);
  }

  function encodeAnswersOneHot(answers, oneHotMap) {
    var vec = new Array(oneHotMap.length);
    for (var i = 0; i < oneHotMap.length; i++) {
      var pair = oneHotMap[i];
      vec[i] = answers[pair[0]] === pair[1] ? 1 : 0;
    }
    return vec;
  }

  function evaluateDecisionTree(node, featureVector) {
    var current = node;
    while (Array.isArray(current)) {
      var featIndex = current[0];
      current = featureVector[featIndex] === 1 ? current[2] : current[1];
    }
    return current;
  }

  function classifyConfidenceTier(confidencePct) {
    if (confidencePct >= 85) {
      return {
        label: 'High',
        className: 'conf-high',
        explanation:
          'Strong agreement across the 100 Random Forest decision trees with clear separation from the 50% decision boundary.'
      };
    }
    if (confidencePct >= 65) {
      return {
        label: 'Moderate',
        className: 'conf-moderate',
        explanation:
          'Majority consensus across the Random Forest ensemble, with some mixed lifestyle or symptom signals.'
      };
    }
    return {
      label: 'Low',
      className: 'conf-low',
      explanation:
        'Near the 50% decision boundary. Your selected inputs contain conflicting signals across the trained model features, resulting in higher uncertainty.'
    };
  }

  function predictWithRandomForest(answers, model) {
    var featureVector = encodeAnswersOneHot(answers, model.oneHotMap);
    var trees = model.trees;
    var treeCount = trees.length;
    var probSum = 0;
    var yesTreeVotes = 0;

    for (var i = 0; i < treeCount; i++) {
      var leafProbYes = evaluateDecisionTree(trees[i], featureVector);
      probSum += leafProbYes;
      if (leafProbYes >= 0.5) {
        yesTreeVotes += 1;
      }
    }

    var rawProbYes = probSum / treeCount;
    var calibratedProbYes = sigmoid(
      model.platt.a * safeLogit(rawProbYes) + model.platt.b
    );

    var isHigherRisk = calibratedProbYes >= 0.5;
    var predictedClass = isHigherRisk ? 'Yes' : 'No';
    var predictionLabel = isHigherRisk ? 'Higher Predicted Risk' : 'Low Predicted Risk';

    var confidenceProb = isHigherRisk ? calibratedProbYes : 1 - calibratedProbYes;
    var confidencePct = Math.round(confidenceProb * 1000) / 10;
    var calibratedYesPct = Math.round(calibratedProbYes * 1000) / 10;
    var rawYesPct = Math.round(rawProbYes * 1000) / 10;
    var alignedTreeCount = isHigherRisk ? yesTreeVotes : treeCount - yesTreeVotes;
    var confidenceTier = classifyConfidenceTier(confidencePct);

    var symptomCount = 0;
    if (answers.frequentUrination === 'Yes') symptomCount += 1;
    if (answers.excessiveThirst === 'Yes') symptomCount += 1;
    if (answers.frequentHunger === 'Yes') symptomCount += 1;
    if (answers.weightChange === 'Yes') symptomCount += 1;

    var description;
    if (isHigherRisk) {
      description =
        'Based on your 9 responses, the trained Random Forest classifier placed this profile in the Higher Predicted Risk group (calibrated confidence: ' +
        confidencePct.toFixed(1) +
        '%, ' +
        confidenceTier.label +
        '). This is an educational model output—please consult a qualified healthcare professional for clinical screening such as HbA1c or fasting glucose testing.';
    } else {
      description =
        'Based on your 9 responses, the trained Random Forest classifier placed this profile in the Low Predicted Risk group (calibrated confidence: ' +
        confidencePct.toFixed(1) +
        '%, ' +
        confidenceTier.label +
        '). Continue maintaining balanced lifestyle habits and consult a healthcare professional if you notice symptoms or have medical concerns.';
    }

    return {
      predictedClass: predictedClass,
      isHigherRisk: isHigherRisk,
      predictionLabel: predictionLabel,
      description: description,
      calibratedProbYes: calibratedProbYes,
      calibratedYesPct: calibratedYesPct,
      rawYesPct: rawYesPct,
      confidencePct: confidencePct,
      confidenceTier: confidenceTier,
      alignedTreeCount: alignedTreeCount,
      treeCount: treeCount,
      symptomCount: symptomCount,
      insights: generateModelObservations(answers, symptomCount, confidenceTier)
    };
  }

  function generateModelObservations(answers, symptomCount, confidenceTier) {
    var observations = [];

    if (answers.highBloodPressure === 'Yes') {
      observations.push(
        'High blood pressure history was reported (Yes). In the study dataset, blood pressure history carries the highest feature weight (63.2% Gini importance) in the Random Forest model.'
      );
    } else {
      observations.push(
        'No history of high blood pressure was reported (No), which strongly aligns with the lower-risk group in the study dataset.'
      );
    }

    if (answers.exerciseDays === 'Never' || answers.exerciseDays === '1-2 Days') {
      observations.push(
        'Weekly exercise frequency was reported as ' +
          answers.exerciseDays +
          '. Exercise frequency is the second strongest predictor in the trained model (32.9% Gini importance); lower activity levels correlate with higher predicted risk in the dataset.'
      );
    } else {
      observations.push(
        'Consistent weekly exercise (' +
          answers.exerciseDays +
          ') was reported, which strongly aligns with the lower-risk pattern in the trained model.'
      );
    }

    if (
      answers.familyHistory === 'Yes (Parents)' ||
      answers.familyHistory === 'Yes (Grandparents)'
    ) {
      observations.push(
        'Family history of diabetes (' +
          answers.familyHistory +
          ') was noted as an important non-modifiable background factor.'
      );
    }

    if (
      answers.craveSweets === 'Often' ||
      answers.craveSweets === 'Very often' ||
      answers.craveSweets === 'Always'
    ) {
      observations.push(
        'Frequent sweet cravings (' +
          answers.craveSweets +
          ') were reported. Monitoring added-sugar intake can support long-term metabolic health.'
      );
    }

    if (symptomCount > 0) {
      observations.push(
        symptomCount +
          ' of 4 classic diabetes symptom checks (urination, thirst, hunger, recent weight change) were answered Yes. Only a qualified clinician and laboratory blood tests can evaluate physical symptoms.'
      );
    } else {
      observations.push(
        'None of the 4 symptom checks (frequent urination, excessive thirst, frequent hunger, or 3-month weight change) were reported.'
      );
    }

    if (confidenceTier.label === 'Low') {
      observations.push(
        'Because your blood pressure and exercise responses point in opposite directions relative to the training dataset, the model reported Low Confidence—indicating genuine classification uncertainty.'
      );
    }

    return observations;
  }

  /* ========================================================================
     5. Result Dashboard Rendering
     ======================================================================== */
  function getResultIconSvg(isHigherRisk) {
    if (isHigherRisk) {
      return (
        '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />' +
        '<line x1="12" y1="9" x2="12" y2="13" />' +
        '<line x1="12" y1="17" x2="12.01" y2="17" />' +
        '</svg>'
      );
    }
    return (
      '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />' +
      '<polyline points="22 4 12 14.01 9 11.01" />' +
      '</svg>'
    );
  }

  function buildSummaryCardsData(answers, prediction) {
    var exerciseActive =
      answers.exerciseDays === '3-4 Days' || answers.exerciseDays === 'Daily';
    var hasBP = answers.highBloodPressure === 'Yes';
    var hasFamily =
      answers.familyHistory === 'Yes (Parents)' ||
      answers.familyHistory === 'Yes (Grandparents)';
    var frequentCravings =
      answers.craveSweets === 'Often' ||
      answers.craveSweets === 'Very often' ||
      answers.craveSweets === 'Always';

    var reportedSymptomsList = [];
    if (answers.frequentUrination === 'Yes') reportedSymptomsList.push('Frequent urination');
    if (answers.excessiveThirst === 'Yes') reportedSymptomsList.push('Excessive thirst');
    if (answers.frequentHunger === 'Yes') reportedSymptomsList.push('Frequent hunger');
    if (answers.weightChange === 'Yes') reportedSymptomsList.push('3-month weight change');

    var exerciseProgress =
      answers.exerciseDays === 'Daily'
        ? 100
        : answers.exerciseDays === '3-4 Days'
          ? 72
          : answers.exerciseDays === '1-2 Days'
            ? 38
            : 14;

    var cravingProgressMap = {
      Never: 12,
      Rarely: 28,
      Sometimes: 46,
      Often: 68,
      'Very often': 84,
      Always: 100
    };

    return [
      {
        label: 'Model Prediction',
        statusText: prediction.confidenceTier.label + ' Confidence',
        tone: prediction.isHigherRisk ? 'tone-review' : 'tone-good',
        value: prediction.predictionLabel,
        detail:
          'Calibrated certainty: ' +
          prediction.confidencePct.toFixed(1) +
          '% (' +
          prediction.alignedTreeCount +
          ' of ' +
          prediction.treeCount +
          ' trees aligned).',
        progress: Math.round(prediction.confidencePct)
      },
      {
        label: 'Weekly Exercise',
        statusText: exerciseActive ? 'Active' : 'Low Activity',
        tone: exerciseActive ? 'tone-good' : 'tone-review',
        value: answers.exerciseDays,
        detail: exerciseActive
          ? 'Regular weekly physical activity aligns with lower risk patterns.'
          : 'Increasing weekly physical activity supports healthier metabolic habits.',
        progress: exerciseProgress
      },
      {
        label: 'Blood Pressure & Age',
        statusText: hasBP ? 'BP Reported' : 'No BP History',
        tone: hasBP ? 'tone-review' : 'tone-good',
        value: 'High BP: ' + answers.highBloodPressure,
        detail: 'Selected age group: ' + answers.ageGroup + '.',
        progress: hasBP ? 88 : 24
      },
      {
        label: 'Family History',
        statusText: hasFamily ? 'Reported' : 'None Reported',
        tone: hasFamily ? 'tone-review' : 'tone-good',
        value: answers.familyHistory,
        detail: 'Included as a hereditary context feature in the 9-feature model.',
        progress: hasFamily ? 76 : 22
      },
      {
        label: 'Sweet Cravings',
        statusText: frequentCravings ? 'Frequent' : 'Moderated',
        tone: frequentCravings ? 'tone-review' : 'tone-good',
        value: answers.craveSweets,
        detail: 'Frequency of sweet food cravings encoded across 6 survey levels.',
        progress: cravingProgressMap[answers.craveSweets] || 50
      },
      {
        label: 'Reported Symptoms',
        statusText:
          reportedSymptomsList.length > 0
            ? reportedSymptomsList.length + ' of 4 Yes'
            : '0 of 4 Yes',
        tone: reportedSymptomsList.length > 0 ? 'tone-review' : 'tone-good',
        value:
          reportedSymptomsList.length > 0
            ? reportedSymptomsList.length + ' symptom(s) noted'
            : 'No symptoms reported',
        detail:
          reportedSymptomsList.length > 0
            ? reportedSymptomsList.join(', ') + '.'
            : 'No urination, thirst, hunger, or 3-month weight changes reported.',
        progress: Math.min(100, Math.max(12, reportedSymptomsList.length * 25))
      }
    ];
  }

  function renderResultDashboard(answers, prediction, model) {
    if (!resultSection) return;

    resultHeroCard.classList.remove('category-low', 'category-moderate', 'category-elevated');
    resultHeroCard.classList.add(prediction.isHigherRisk ? 'category-elevated' : 'category-low');

    resultStatusIcon.innerHTML = getResultIconSvg(prediction.isHigherRisk);
    resultCategoryTitle.textContent = prediction.predictionLabel;
    resultCategoryDesc.textContent = prediction.description;

    /* Confidence Panel */
    confidencePanel.classList.remove('conf-high', 'conf-moderate', 'conf-low');
    confidencePanel.classList.add(prediction.confidenceTier.className);

    confidencePercentage.textContent = prediction.confidencePct.toFixed(1) + '%';
    confidenceCategoryLabel.textContent =
      prediction.confidenceTier.label + ' Confidence';
    confidenceTreesMeta.textContent =
      prediction.alignedTreeCount + ' / ' + prediction.treeCount + ' trees';

    var barWidth = Math.max(4, Math.min(100, prediction.confidencePct));
    confidenceBarFill.style.width = barWidth + '%';
    confidenceBarTrack.setAttribute('aria-valuenow', String(prediction.confidencePct.toFixed(1)));

    probYesVal.textContent = prediction.calibratedYesPct.toFixed(1) + '%';
    rawProbYesVal.textContent = prediction.rawYesPct.toFixed(1) + '%';
    confidenceNote.textContent =
      'Note: ' +
      prediction.confidenceTier.explanation +
      ' This score represents the calibrated model’s estimated certainty on the study dataset—it is not a medical diagnosis or a guarantee of accuracy.';

    /* Probability Spectrum */
    var markerPos = Math.max(2, Math.min(98, prediction.calibratedYesPct));
    resultScaleMarker.style.left = markerPos + '%';
    if (spectrumReadout) {
      spectrumReadout.textContent =
        'P(Higher Risk) = ' +
        prediction.calibratedYesPct.toFixed(1) +
        '% · Decision Threshold = 50.0%';
    }

    /* Live Model Evaluation Metrics from exported model-data.js */
    if (modelMetricsText && model && model.evaluation && model.evaluation.testCalibrated) {
      var ev = model.evaluation.testCalibrated;
      modelMetricsText.textContent =
        'Accuracy: ' +
        (ev.accuracy * 100).toFixed(2) +
        '% · Precision: ' +
        (ev.precision * 100).toFixed(2) +
        '% · Recall: ' +
        (ev.recall * 100).toFixed(2) +
        '% · F1: ' +
        (ev.f1 * 100).toFixed(2) +
        '% · ROC-AUC: ' +
        ev.rocAuc.toFixed(4) +
        ' · Brier: ' +
        ev.brierScore.toFixed(4) +
        ' · ECE: ' +
        (ev.ece * 100).toFixed(2) +
        '% (5-fold Platt sigmoid scaling).';
    }

    /* Summary Cards */
    var cardsData = buildSummaryCardsData(answers, prediction);
    resultCardsGrid.innerHTML = '';

    cardsData.forEach(function (card, index) {
      var cardEl = document.createElement('article');
      cardEl.className = 'summary-card';
      cardEl.style.setProperty('--card-index', String(index));

      cardEl.innerHTML =
        '<div class="summary-card-head">' +
        '<span class="summary-card-label"></span>' +
        '<span class="summary-status-text ' +
        card.tone +
        '"></span>' +
        '</div>' +
        '<div class="summary-card-value"></div>' +
        '<p class="summary-card-detail"></p>' +
        '<div class="mini-progress-bar"><div class="mini-progress-fill" style="width:' +
        card.progress +
        '%"></div></div>';

      cardEl.querySelector('.summary-card-label').textContent = card.label;
      cardEl.querySelector('.summary-status-text').textContent = card.statusText;
      cardEl.querySelector('.summary-card-value').textContent = card.value;
      cardEl.querySelector('.summary-card-detail').textContent = card.detail;

      resultCardsGrid.appendChild(cardEl);
    });

    /* Observations List */
    insightsList.innerHTML = '';
    prediction.insights.forEach(function (text) {
      var li = document.createElement('li');
      li.innerHTML =
        '<span class="insight-bullet" aria-hidden="true">' +
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<polyline points="20 6 9 17 4 12" />' +
        '</svg>' +
        '</span>' +
        '<span class="insight-text"></span>';
      li.querySelector('.insight-text').textContent = text;
      insightsList.appendChild(li);
    });

    assessmentShell.hidden = true;
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function handleAssessmentSubmission() {
    var model = window.DIABETES_RF_MODEL;
    if (!model || !Array.isArray(model.trees) || !Array.isArray(model.oneHotMap)) {
      showCustomValidationError(
        'The prediction model data (model-data.js) could not be loaded. Please refresh the page and try again.',
        []
      );
      return;
    }

    var check = validateAllAnswersAgainstModel(state.answers, model);
    if (!check.valid) {
      showCustomValidationError(check.message, check.missingFields);
      return;
    }

    try {
      var prediction = predictWithRandomForest(state.answers, model);
      renderResultDashboard(state.answers, prediction, model);
    } catch (err) {
      showCustomValidationError(
        'An unexpected error occurred while evaluating the model. Please verify your selections and try again.',
        []
      );
    }
  }

  /* ========================================================================
     6. Result Actions (Retake & Edit Answers)
     ======================================================================== */
  function initResultActions() {
    if (retakeBtn) {
      retakeBtn.addEventListener('click', function () {
        state.answers = {};
        if (assessmentForm) {
          assessmentForm.reset();
        }
        resultSection.hidden = true;
        assessmentShell.hidden = false;
        renderStep(1, 'backward');
        assessmentShell.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }

    if (editAnswersBtn) {
      editAnswersBtn.addEventListener('click', function () {
        resultSection.hidden = true;
        assessmentShell.hidden = false;
        renderStep(1, 'backward');
        assessmentShell.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  }

  /* ========================================================================
     Bootstrap
     ======================================================================== */
  document.addEventListener('DOMContentLoaded', function () {
    initNavigation();
    initScrollReveal();
    initQuestionnaire();
    initResultActions();
  });
})();

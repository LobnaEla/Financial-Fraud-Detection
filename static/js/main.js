document.addEventListener('DOMContentLoaded', function () {
    // Configuration de l'API - Mise à jour des URLs pour correspondre à la structure du blueprint
    const API_BASE_URL = 'http://127.0.0.1:5000';
    const API_STATUS_URL = `${API_BASE_URL}/status`;
    const API_PREDICT_URL = `${API_BASE_URL}/api/predict`;

    // Vérification de l'état de l'API
    checkApiStatus();

    // Gestionnaire d'onglets
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));

            const tabName = tab.getAttribute('data-tab');
            if (tabName === 'csv') {
                document.getElementById('csvTab').classList.add('active');
            } else {
                document.getElementById('jsonTab').classList.add('active');
            }
        });
    });

    // Ajouter un clic sur l'onglet CSV au chargement de la page
    const csvTab = document.querySelector('.tab[data-tab="csv"]');
    if (csvTab) {
        csvTab.click();
    }

    // Fonction pour remplacer les NaN par null dans un objet JSON
    function replaceNaN(obj) {
        for (let key in obj) {
            // Vérification si c'est NaN, la chaîne 'NaN' ou Infinity
            if (obj[key] !== obj[key] ||
                obj[key] === 'NaN' ||
                obj[key] === NaN ||
                obj[key] === Infinity ||
                obj[key] === -Infinity ||
                obj[key] === 'Infinity' ||
                obj[key] === '-Infinity') {
                obj[key] = null;
            } else if (typeof obj[key] === 'object' && obj[key] !== null) {
                replaceNaN(obj[key]);
            }
        }
        return obj;
    }

    // Fonction pour corriger un texte JSON contenant des valeurs non supportées
    function sanitizeJsonText(jsonText) {
        // Corriger les valeurs non valides en JSON
        return jsonText
            .replace(/:\s*NaN/g, ': null')
            .replace(/:\s*Infinity/g, ': null')
            .replace(/:\s*-Infinity/g, ': null')
            .replace(/"\s*:\s*NaN/g, '": null')
            .replace(/"\s*:\s*Infinity/g, '": null')
            .replace(/"\s*:\s*-Infinity/g, '": null');
    }

    // Gestionnaire du JSON
    document.getElementById('submitJson').addEventListener('click', function () {
        try {
            const jsonFile = document.getElementById("jsonFile").files[0];
            let jsonText = '';

            if (jsonFile) {
                // Si un fichier JSON est sélectionné
                const reader = new FileReader();

                reader.onload = function (event) {
                    jsonText = event.target.result;
                    handleJsonParsing(jsonText);
                };

                reader.onerror = function (error) {
                    showResult('error', 'Erreur de lecture', 'Erreur lors de la lecture du fichier JSON: ' + error.message);
                };

                // Lire le fichier JSON
                reader.readAsText(jsonFile);
            } else {
                // Si aucun fichier n'est sélectionné, on prend le contenu du textarea
                jsonText = document.getElementById('jsonInput').value.trim();

                // Si le textarea est vide, afficher une erreur
                if (!jsonText) {
                    showResult('error', 'Erreur', 'Le contenu JSON est vide.');
                    return;
                }

                handleJsonParsing(jsonText);
            }
        } catch (error) {
            showResult('error', 'Erreur JSON', 'Le format JSON n\'est pas valide: ' + error.message);
        }
    });

    // Fonction de traitement de la chaîne JSON
    function handleJsonParsing(jsonText) {
        // Nettoyer le JSON (en cas de NaN ou autres problèmes)
        const sanitizedJsonText = sanitizeJsonText(jsonText);

        console.log("JSON original:", jsonText);
        console.log("JSON nettoyé:", sanitizedJsonText);

        try {
            const jsonData = JSON.parse(sanitizedJsonText);
            console.log("Envoi des données JSON:", jsonData);
            sendRequest(jsonData);
        } catch (parseError) {
            showResult('error', 'Erreur JSON', 'Le format JSON n\'est pas valide même après nettoyage: ' + parseError.message);
            console.error("Erreur de parsing JSON:", parseError);
            console.log("JSON problématique:", sanitizedJsonText);
        }
    }

    // Gestionnaire du CSV
    function csvToJson(csv) {
        const lines = csv.trim().split("\n");
        if (lines.length < 2) throw new Error("Le CSV doit contenir au moins deux lignes.");
        const headers = lines[0].split(",");
        const values = lines[1].split(",");
        if (headers.length !== values.length) throw new Error("Nombre de colonnes incorrect.");

        const obj = {};
        headers.forEach((h, i) => {
            const value = values[i].trim();
            // Convertir en numérique si possible
            const numericValue = !isNaN(value) ? parseFloat(value) : value;
            obj[h.trim()] = numericValue;
        });
        return obj;
    }

    document.getElementById('submitCsv').addEventListener('click', function () {
        try {
            const csvFile = document.getElementById("csvFile").files[0];
            let csvText = '';

            if (csvFile) {
                // Si un fichier CSV est sélectionné
                const reader = new FileReader();

                reader.onload = function (event) {
                    csvText = event.target.result;
                    handleCsvParsing(csvText);
                };

                reader.onerror = function (error) {
                    showResult('error', 'Erreur de lecture', 'Erreur lors de la lecture du fichier CSV: ' + error.message);
                };

                // Lire le fichier CSV
                reader.readAsText(csvFile);
            } else {
                // Si aucun fichier n'est sélectionné, on prend le contenu du textarea
                csvText = document.getElementById('csvInput').value.trim();

                // Si le textarea est vide, afficher une erreur
                if (!csvText) {
                    showResult('error', 'Erreur', 'Le contenu CSV est vide.');
                    return;
                }

                handleCsvParsing(csvText);
            }
        } catch (error) {
            showResult('error', 'Erreur CSV', 'Le format CSV n\'est pas valide: ' + error.message);
        }
    });

    // Fonction de traitement du CSV
    function handleCsvParsing(csvText) {
        try {
            const jsonData = csvToJson(csvText);
            console.log("Données JSON issues du CSV:", jsonData);
            sendRequest(jsonData);
        } catch (parseError) {
            showResult('error', 'Erreur CSV', 'Le format CSV n\'est pas valide: ' + parseError.message);
            console.error("Erreur de parsing CSV:", parseError);
            console.log("CSV problématique:", csvText);
        }
    }

    window.clearCSVSection = function () {
        document.getElementById('csvFile').value = '';
        document.getElementById('csvInput').value = '';
    };

    window.clearJSONSection = function () {
        document.getElementById('jsonFile').value = '';
        document.getElementById('jsonInput').value = '';
    };


    // async function handlePrediction() {
    //     const resultDiv = document.getElementById("result");
    //     resultDiv.innerHTML = "⏳ Envoi de la transaction...";
    //     resultDiv.style.color = "black";

    //     try {
    //         let jsonData = null;

    //         // Vérifie si fichier JSON est fourni
    //         const jsonFile = document.getElementById("jsonFile").files[0];
    //         if (jsonFile) {
    //             const text = await jsonFile.text();
    //             jsonData = JSON.parse(text);
    //         }
    //         else {
    //             // Vérifie si fichier CSV est fourni
    //             const csvFile = document.getElementById("csvFile").files[0];
    //             if (csvFile) {
    //                 const text = await csvFile.text();
    //                 jsonData = csvToJson(text);
    //             }
    //             else {
    //                 // Sinon, utilise le contenu du textarea CSV
    //                 const csvText = document.getElementById("csvInput").value;
    //                 jsonData = csvToJson(csvText);
    //             }
    //         }

    //         const response = await fetch('/predict', {
    //             method: 'POST',
    //             headers: { 'Content-Type': 'application/json' },
    //             body: JSON.stringify(jsonData)
    //         });

    //         if (!response.ok) {
    //             const err = await response.json().catch(() => ({}));
    //             const message = err.error || response.statusText || 'Erreur inconnue';
    //             resultDiv.innerHTML = `⚠️ Erreur du serveur : ${message}`;
    //             resultDiv.style.color = 'orange';
    //             return;
    //         }

    //         const result = await response.json();
    //         if (result.isFraud === 1) {
    //             resultDiv.innerHTML = "❌ Fraude détectée !";
    //             resultDiv.style.color = "red";
    //         } else {
    //             resultDiv.innerHTML = "✅ Transaction normale.";
    //             resultDiv.style.color = "green";
    //         }

    //     } catch (error) {
    //         resultDiv.innerHTML = `⚠️ Erreur : ${error.message}`;
    //         resultDiv.style.color = "orange";
    //     }
    // }

    // Fonction pour vérifier l'état de l'API
    function checkApiStatus() {
        const statusElement = document.getElementById('apiStatus');

        fetch(API_STATUS_URL)
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('Erreur de réponse');
            })
            .then(data => {
                if (data.status === 'online') {
                    statusElement.textContent = 'API en ligne';
                    statusElement.className = 'api-status api-online';
                } else {
                    statusElement.textContent = 'API en ligne mais état inconnu';
                    statusElement.className = 'api-status api-offline';
                }
            })
            .catch(error => {
                statusElement.textContent = 'API hors ligne';
                statusElement.className = 'api-status api-offline';
                console.error('Erreur API:', error);
            });
    }

    // Fonction pour envoyer la requête
    function sendRequest(data) {
        // Remplacer les NaN par null avant l'envoi
        data = replaceNaN(data);

        // Afficher le spinner
        document.querySelector('.loading').style.display = 'block';
        document.getElementById('result').style.display = 'none';

        console.log(`Envoi de la requête à ${API_PREDICT_URL}`);
        console.log("Données envoyées:", JSON.stringify(data));

        fetch(API_PREDICT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
            .then(response => {
                console.log("Réponse reçue:", response);
                if (!response.ok) {
                    throw new Error(`Erreur réseau (${response.status}): ${response.statusText}`);
                }
                return response.json();
            })
            .then(result => {
                // Masquer le spinner
                document.querySelector('.loading').style.display = 'none';
                console.log("Résultat de la prédiction:", result);

                if (result.fraud_prediction === 1) {
                    showResult('fraud', 'Transaction suspecte détectée',
                        'Cette transaction a été identifiée comme potentiellement frauduleuse.');
                } else {
                    showResult('legitimate', 'Transaction légitime',
                        'Aucune activité suspecte n\'a été détectée dans cette transaction.');
                }
            })
            .catch(error => {
                // Masquer le spinner
                document.querySelector('.loading').style.display = 'none';
                console.error("Erreur:", error);
                showResult('error', 'Erreur', 'Une erreur s\'est produite: ' + error.message);
            });
    }

    // Fonction pour afficher le résultat
    function showResult(type, title, message) {
        const resultDiv = document.getElementById('result');
        resultDiv.className = 'result ' + type;
        document.getElementById('resultTitle').textContent = title;
        document.getElementById('resultMessage').textContent = message;
        resultDiv.style.display = 'block';
    }

});
pipeline {
    agent any

    triggers {
        // Trigger Jenkins build on GitHub push event
        githubPush()
    }

    stages {
        stage('Checkout') {
            steps {
                script {
                    // Checkout from GitHub with specified branch and credentials
                    git branch: 'main', credentialsId: 'github-credentials', url: 'https://github.com/udaybhadauria/AI-CSV-Files-Chatbot.git'
                }
            }
        }

        stage('Test Trigger') {
            steps {
                echo "GitHub webhook triggered Jenkins build successfully!"
            }
        }
    }

    post {
        success {
            echo '✅ Jenkins Build Triggered Successfully!'
        }
        failure {
            echo '❌ Jenkins Build Failed. Check logs.'
        }
    }
}
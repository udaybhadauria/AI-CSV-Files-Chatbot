pipeline {
    agent any

    triggers {
        githubPush()  // This ensures Jenkins triggers on a GitHub push event
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', credentialsId: 'github-credentials', url: 'https://github.com/udaybhadauria/AI-CSV-Files-Chatbot.git' 
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
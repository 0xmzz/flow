plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.flow.bubble"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.flow.bubble"
        minSdk = 26
        targetSdk = 34
        versionCode = 8
        versionName = "0.8"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core:1.13.1")  // FileProvider for self-update
}

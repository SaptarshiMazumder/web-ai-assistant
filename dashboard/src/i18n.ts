import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import enTranslations from './locales/en.json'
import jaTranslations from './locales/ja.json'

// Configure i18next
i18n
    .use(LanguageDetector)
    .use(initReactI18next) // pass the i18n instance to react-i18next.
    .init({
        resources: {
            en: {
                translation: enTranslations
            },
            ja: {
                translation: jaTranslations
            }
        },
        fallbackLng: 'en', // fallback language if translation not found

        detection: {
            // Order of checks: URL -> LocalStorage -> Browser Settings
            order: ['querystring', 'localStorage', 'navigator'],
            // Where to cache the user's explicit choice
            caches: ['localStorage'],
        },

        interpolation: {
            escapeValue: false // react already safes from xss
        }
    })

export default i18n

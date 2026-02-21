import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import enTranslations from './locales/en.json'
import jaTranslations from './locales/ja.json'

// Configure i18next
i18n
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
        lng: 'en', // default language
        fallbackLng: 'en', // fallback language if translation not found

        interpolation: {
            escapeValue: false // react already safes from xss
        }
    })

export default i18n

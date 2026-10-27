import unittest

import numpy as np

from core.global_game_state import GlobalGameState
from core.tables.weather import WeatherResult, WeatherType

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    WEATHER_START,
    WEATHER_END,
    WEATHER_CLEAR_INDEX,
    WEATHER_PARTLY_CLEAR_INDEX,
    WEATHER_OVERCAST_INDEX,
    encode_weather,
)


class TestEncodeWeather(unittest.TestCase):

    def setUp(self):
        GlobalGameState.current_weather = None

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_weather_none(self):
        encode_weather(self.observation)

        weather_state = self.observation[
            WEATHER_START:WEATHER_END
        ]

        self.assertEqual(
            np.count_nonzero(weather_state),
            0,
        )

    def test_encode_weather_clear(self):
        GlobalGameState.current_weather = WeatherResult(
            weather_type=WeatherType.CLEAR,
            available_jabos=999,
            resource_drm=0,
            carpet_bombing_drm=0,
        )

        encode_weather(self.observation)

        self.assertEqual(
            self.observation[WEATHER_CLEAR_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[WEATHER_PARTLY_CLEAR_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[WEATHER_OVERCAST_INDEX],
            0.0,
        )

    def test_encode_weather_partly_clear(self):
        GlobalGameState.current_weather = WeatherResult(
            weather_type=WeatherType.PARTLY_CLEAR,
            available_jabos=1,
            resource_drm=0,
            carpet_bombing_drm=1,
        )

        encode_weather(self.observation)

        self.assertEqual(
            self.observation[WEATHER_CLEAR_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[WEATHER_PARTLY_CLEAR_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[WEATHER_OVERCAST_INDEX],
            0.0,
        )

    def test_encode_weather_overcast(self):
        GlobalGameState.current_weather = WeatherResult(
            weather_type=WeatherType.OVERCAST,
            available_jabos=0,
            resource_drm=1,
            carpet_bombing_drm=-999,
        )

        encode_weather(self.observation)

        self.assertEqual(
            self.observation[WEATHER_CLEAR_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[WEATHER_PARTLY_CLEAR_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[WEATHER_OVERCAST_INDEX],
            1.0,
        )

    def test_encode_weather_only_changes_weather_block(self):
        GlobalGameState.current_weather = WeatherResult(
            weather_type=WeatherType.CLEAR,
            available_jabos=999,
            resource_drm=0,
            carpet_bombing_drm=0,
        )

        encode_weather(self.observation)

        before_weather = self.observation[:WEATHER_START]
        after_weather = self.observation[WEATHER_END:]

        self.assertEqual(
            np.count_nonzero(before_weather),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_weather),
            0,
        )


if __name__ == "__main__":
    unittest.main()

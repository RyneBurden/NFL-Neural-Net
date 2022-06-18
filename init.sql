CREATE TABLE predictions
                        (
                            id          SERIAL  PRIMARY KEY,
                            week        INTEGER            ,
                            season      INTEGER            ,
                            home_team   TEXT               ,
                            away_team   TEXT               ,
                            winner      VARCHAR(4)          -- This will be either 'home' or 'away'
                        );
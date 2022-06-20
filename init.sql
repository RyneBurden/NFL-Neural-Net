CREATE TABLE predictions
                        (
                            id          SERIAL  PRIMARY KEY,
                            week        SMALLINT            ,
                            season      SMALLINT           ,
                            home_team   TEXT               ,
                            away_team   TEXT               ,
                            winner      VARCHAR(4)          -- This will be either 'home' or 'away'
                        );

CREATE TABLE teams_logos_colors
                        (
                            team_abbr           VARCHAR(3) PRIMARY KEY,
                            team_name           TEXT                  ,
                            team_id             SMALLINT              ,
                            team_nick           TEXT                  ,
                            team_color          VARCHAR(7)            ,
                            team_color2         VARCHAR(7)            ,
                            team_color3         VARCHAR(7)            ,
                            team_color4         VARCHAR(7)            ,
                            team_logo_wikipedia TEXT                  ,
                            team_logo_espn      TEXT                  ,
                            team_wordmark       TEXT
                        );

COPY teams_logos_colors 
FROM '/staley/teams_logos_colors.csv'
DELIMITER ',' CSV HEADER;
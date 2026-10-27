ottomanDeckGUID = "4cb217"
wahhabiDeckGUID = "e6a988"

shuffleUsed = false
originalY = nil

HIDE_DEPTH = 2.0


function onSave()
    return JSON.encode({
        shuffleUsed = shuffleUsed,
        originalY = originalY,
        ottomanDeckGUID = ottomanDeckGUID,
        wahhabiDeckGUID = wahhabiDeckGUID
    })
end


function onLoad(saved_data)

    -- Restore saved information
    if saved_data ~= nil and saved_data ~= "" then
        local data = JSON.decode(saved_data)

        if data then

            if data.shuffleUsed ~= nil then
                shuffleUsed = data.shuffleUsed
            end

            if data.originalY ~= nil then
                originalY = data.originalY
            end



        end
    end

    -- First time only: remember where this object normally sits
    if originalY == nil then
        originalY = self.getPosition().y
    end

    self.setLock(true)

    -- Put the physical black tile at a FIXED hidden height
    local p = self.getPosition()

    self.setPosition({
        p.x,
        originalY - HIDE_DEPTH,
        p.z
    })

    createButtons()
end


function createButtons()

    self.clearButtons()

    -- Because the physical host is 2 units lower,
    -- raise the scripted button by exactly the same amount.
    local buttonY = 0.6 + HIDE_DEPTH

    if not shuffleUsed then

        self.createButton({
            label = "Shuffle",
            click_function = "shuffleDecks",
            function_owner = self,
            position = {3, buttonY, 0},
            rotation = {0, 0, 0},
            width = 1200,
            height = 500,
            font_size = 250,
            color = {1, 1, 1},
            font_color = {0, 0, 0}
        })

    else

        self.createButton({
            label = "Draw",
            click_function = "drawCards",
            function_owner = self,
            position = {3, buttonY, 0},
            rotation = {0, 0, 0},
            width = 1200,
            height = 500,
            font_size = 250,
            color = {1, 1, 1},
            font_color = {0, 0, 0}
        })

    end
end


function getDeck(guid)
    return getObjectFromGUID(guid)
end


function drawCardFromDeck(guid, offset, deckType)

    local deck = getObjectFromGUID(guid)

    if not deck then
        return
    end

    local deckPos = deck.getPosition()

    -- Draw the card above the table/deck so it cannot
    -- accidentally slide underneath the pile.
    local pos = {
        deckPos.x + offset[1],
        deckPos.y + 3.0,
        deckPos.z + offset[3]
    }


    if deck.tag == "Deck" then

        deck.takeObject({
            position = pos,
            rotation = {0, 180, 0},
            smooth = true,

            callback_function = function(card)

                -- TTS converts a two-card deck into a single Card
                -- after one of the cards is removed.
                -- That remaining Card can receive a new GUID.
                --
                -- Wait briefly for TTS to finish the conversion,
                -- then find the remaining Deck/Card at the
                -- original deck position and remember its GUID.

                Wait.frames(function()

                    local objects = Physics.cast({
                        origin = {
                            deckPos.x,
                            deckPos.y - 1,
                            deckPos.z
                        },
                        direction = {0, 1, 0},
                        type = 3,
                        size = {2, 4, 2},
                        max_distance = 4
                    })

                    for _, hit in ipairs(objects) do

                        local obj = hit.hit_object

                        if obj
                        and obj ~= card
                        and (obj.tag == "Deck" or obj.tag == "Card") then

                            if deckType == "ottoman" then
                                ottomanDeckGUID = obj.getGUID()
                            elseif deckType == "wahhabi" then
                                wahhabiDeckGUID = obj.getGUID()
                            end

                            break
                        end

                    end

                end, 3)
            end
        })


    elseif deck.tag == "Card" then

        -- Only one card remains.
        -- Move that card directly to the draw position.

        deck.setPositionSmooth(pos)
        deck.setRotationSmooth({0, 180, 0})

    end
end


function drawCards()

    drawCardFromDeck(
        ottomanDeckGUID,
        {0, 0, -8},
        "ottoman"
    )

    drawCardFromDeck(
        wahhabiDeckGUID,
        {0, 0, -8},
        "wahhabi"
    )
end


function shuffleDecks()

    local ottomanDeck =
        getDeck(ottomanDeckGUID)

    local wahhabiDeck =
        getDeck(wahhabiDeckGUID)


    if ottomanDeck
    and ottomanDeck.tag == "Deck" then
        ottomanDeck.shuffle()
    end


    if wahhabiDeck
    and wahhabiDeck.tag == "Deck" then
        wahhabiDeck.shuffle()
    end


    shuffleUsed = true

    createButtons()
end
--[[ Lua code. See documentation: https://api.tabletopsimulator.com/ --]]

--[[ The onLoad event is called after the game save finishes loading. --]]
function onLoad()
    --[[ print('onLoad!') --]]
end

--[[ The onUpdate event is called once per frame. --]]
function onUpdate()
    --[[ print('onUpdate loop!') --]]
end

function onScriptingButtonDown(index, playerColor)

    -- Scripting 1: RED DIE
    if index == 1 then
        local redDie = getObjectFromGUID("b4b33e")

        if redDie then
            redDie.randomize()
        end

    -- Scripting 2: GREEN DIE
    elseif index == 2 then
        local greenDie = getObjectFromGUID("951739")

        if greenDie then
            greenDie.randomize()
        end

    -- Scripting 3: BOTH DICE
    elseif index == 3 then
        local redDie = getObjectFromGUID("b4b33e")
        local greenDie = getObjectFromGUID("951739")

        if redDie then
            redDie.randomize()
        end

        if greenDie then
            greenDie.randomize()
        end
    end

end
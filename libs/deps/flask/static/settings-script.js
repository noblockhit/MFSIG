function isNumeric(str) {
    if (typeof str != "string") return false; // we only process strings!
    return (
        (!isNaN(str) && // use type coercion to parse the _entirety_ of the string (`parseFloat` alone does not do this)...
            !isNaN(parseFloat(str))) || // ...and ensure strings of whitespace fail
        str === ""
    ); // except completly empty to ease the retyping of the first digit
}

function prevent_submit_and_unfocus(e) {
    document.activeElement.blur()
    e.preventDefault();
}

const getUrIParameter = function getUrlParameter(sParam) {
    var sPageURL = window.location.search.substring(1),
        sURLVariables = sPageURL.split('&'),
        sParameterName,
        i;

    for (i = 0; i < sURLVariables.length; i++) {
        sParameterName = sURLVariables[i].split('=');

        if (sParameterName[0] === sParam) {
            return sParameterName[1] === undefined ? true : decodeURIComponent(sParameterName[1]);
        }
    }
    return false;
};

$(window).on("load", () => {
    $("form").submit(prevent_submit_and_unfocus);
    
    $(".submit-button").on("click", function () {
        const attr_id =  $(this).attr("for")
        
        const id_name_table = {
            "gpio-default-on-input": "GPIO-default-on",
            "gpio-motor-pins-input": "GPIO-motor-pins",
            "gpio-camera-pin-input": "GPIO-camera-pin"
        }

        const attr_name = id_name_table[attr_id];

        let value = $(`#${attr_id}`).val();
        if (attr_id === "gpio-motor-pins-input") {
            value = `[${value}]`
        }

        $.post(`/settings/${attr_name}/${value}`);
    })
    
    
    // load saved values

    $.get("/settings/GPIO-default-on", (async=false), (value) => {
        $("#gpio-default-on-input").val(value);
    });

    $.get("/settings/GPIO-motor-pins", (async=false), (value) => {
        console.log(value)
        $("#gpio-motor-pins-input").val(value.substring(1, value.length-1));
    });

    $.get("/settings/GPIO-camera-pin", (async=false), (value) => {
        $("#gpio-camera-pin-input").val(parseInt(value))
    });
});

console.log("here")
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
    temp_total_steps = getUrIParameter("total-steps")
    if (temp_total_steps === false) {
        temp_total_steps = 800;
    }
    $("#total-steps").text(temp_total_steps)

    let progressBar = document.getElementById("recording-progress-beam");
    let progressNum = document.getElementById("progress-value-num")
    progressBar.style.width = "0%";

    var socket = new WebSocket(`ws://${window.location.hostname}:65432`);

    socket.addEventListener("open", () => {
        // send a message to the server
        socket.send(null)
        console.log("connected")
    });

    // receive a message from the server
    socket.addEventListener("message", ({ data }) => {
        console.log("recv", data)
        progressBar.style.width = `${data}%`;
        progressNum.innerHTML = `${data}%`
        socket.send(data);
    });

    var idd_input = document.getElementById("image-delta-distance-input");
    var tia_input = document.getElementById("total-image-amount-input");
    var dpr_input = document.getElementById("distance-per-rotation-input");
    var mspr_input = document.getElementById("motor-steps-per-rotation-input");

    $("#image-delta-distance-form").submit(prevent_submit_and_unfocus);
    $("#total-image-amount-form").submit(prevent_submit_and_unfocus);
    $("#distance-per-rotation-form").submit(prevent_submit_and_unfocus);
    $("#motor-steps-per-rotation-form").submit(prevent_submit_and_unfocus);

    var idd_value = idd_input.value;
    var tia_value = tia_input.value;
    var dpr_value = dpr_input.value;
    var mspr_value = mspr_input.value;

    var total_steps = parseFloat($("#total-steps").text())

    var latest_typed = tia_input;

    function update_all() {
        console.log("update triggered")

        distance_per_step = dpr_value / mspr_value;
        total_distance = total_steps * distance_per_step;

        if (latest_typed === idd_input) {
            tia_value = total_distance / idd_value;
            tia_input.value = tia_value;
        }

        if (latest_typed === tia_input) {
            idd_value = total_distance / tia_value;
            idd_input.value = idd_value;
        }

        if (tia_value != Infinity) {
            $.post(`/image-count/${tia_value}`);
        }

        $.post(`/settings/steps-per-motor-rotation/${mspr_value}`);
        $.post(`/settings/distance-per-motor-rotation/${dpr_value}`);
    }

    $(idd_input).on("keyup", function (e) {
        if (isNumeric(idd_input.value) === false) {
            idd_input.value = idd_value;
        } else {
            if (idd_input.value === "") {
            } else {
                idd_value = parseFloat(idd_input.value);
            }
        }

        latest_typed = idd_input;
        update_all()
    });

    $(tia_input).on("keyup", function (e) {
        if (isNumeric(tia_input.value) === false) {
            tia_input.value = tia_value;
        } else {
            if (tia_input.value === "") {
            } else {
                tia_value = parseFloat(tia_input.value);
            }
        }

        latest_typed = tia_input;
        update_all()
    });

    $(dpr_input).on("keyup", function (e) {
        if (isNumeric(dpr_input.value) === false) {
            dpr_input.value = dpr_value;
        } else {
            if (dpr_input.value === "") {
            } else {
                dpr_value = parseFloat(dpr_input.value);
            }
        }
    
        update_all()
    });

    $(mspr_input).on("keyup", function (e) {
        if (isNumeric(mspr_input.value) === false) {
            mspr_input.value = mspr_value;
        } else {
            if (mspr_input.value === "") {
            } else {
                mspr_value = parseFloat(mspr_input.value);
            }
        }

        update_all()
    });

    $("#record-images").on("pointerdown", () => {
        $.get("/record-images", (async = false), () => {
            $("#record-images").html("Started recording!")
        });
    });

    $("#units-input").on("change", function () {
        $.post(`/settings/motor-rotation-units/${Math.log10(this.value)}`);
        $(".unit").html($("#units-input option:selected").text());
    });


    // load saved values

    $.get("/settings/motor-rotation-units", (async = false), (value) => {
        $("#units-input").val(Math.pow(10, parseInt(value)), value)
        $(".unit").html($("#units-input option:selected").text());
    });

    $.get("/settings/steps-per-motor-rotation", (async = false), (value) => {
        mspr_value = parseFloat(value);
        update_all();
        $("#motor-steps-per-rotation-input").val(parseFloat(value))
    });

    $.get("/settings/distance-per-motor-rotation", (async = false), (value) => {
        dpr_value = parseFloat(value);
        update_all();
        $("#distance-per-rotation-input").val(parseFloat(value))
    });
});

function removeFadeOut( el ) {
    const domRect = el.getBoundingClientRect();
    el.style.top = `-${domRect.y + domRect.height}px`;
    el.style.opacity = 0;
    setTimeout(function() {
        el.parentNode.removeChild(el);
    }, 1000);
}

function popup(title, message, container_elem, color) {
    const el = $(`<div style="width: ${$(container_elem).width() - 2* parseFloat($("body").css("font-size"))}px; background-color:${color}", class="popup-container">
    <button class="close-button" aria-label="Close alert" type="button" data-close>
        <span aria-hidden="true">&times;</span>
    </button>
    <h2>${title}</h2>
    <h3>${message}</h3>
    </div>`);

    $(container_elem).append(el)
    
    el.on("pointerup", function(event) {
        // $(event.target).closest(".popup-container").remove()
        removeFadeOut($(event.target).closest(".popup-container")[0])
    })
    
}

$(document).ajaxError(function myErrorHandler(
    event,
    xhr,
    ajaxOptions,
    thrownError
) {
    const err = xhr.responseText;
    const container = $(".container")[0]
    popup("An error has occured", err, container, "rgb(255, 0, 0)")
});


$(document).on("ajaxSuccess", function myMiscHandler(
    event,
    xhr,
    ajaxOptions,
    thrownError
) {

    if (xhr.status == 299) {
        const warning = xhr.responseText;
        const container = $(".container")[0]
        popup("Warning!", warning, container, "rgb(255, 255, 0)")

    }
});